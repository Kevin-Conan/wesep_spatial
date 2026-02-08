import torch
import torch.nn as nn
import torch.nn.functional as F

from wesep.modules.spatial.spatial_frontend import SpatialFrontend,CycEncoder
from wesep.modules.separator.nbc2 import NBC2
from wesep.modules.common.deep_update import deep_update

class TSE_NBC2_SPATIAL_EMB(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # --- 1. Basic Configs ---
        self.win = config.get("win",512)
        self.stride = config.get("stride",256)
        
        self.window = torch.hann_window(self.win)
        
        freq_bins = self.win // 2 + 1
        
        # --- 2. Spatial Configs ---
        spatial_configs = {
            "geometry": {
                # [关键] 确保这里的 n_fft 与 STFT 实际使用的参数一致
                "n_fft": self.win,              
                "fs": 16000,
                "c": 343.0,
                "mic_spacing": 0.03333333,
                "mic_coords": [
                    [-0.05,        0.0, 0.0],  # Mic 0
                    [-0.01666667,  0.0, 0.0],  # Mic 1
                    [ 0.01666667,  0.0, 0.0],  # Mic 2
                    [ 0.05,        0.0, 0.0],  # Mic 3
                ],
            },
            "pairs": [
                [0, 1], [1, 2], [2, 3], [0, 3]
            ],
            "features": {
                "ipd": {"enabled": False},
                "cdf": {"enabled": False},
                "sdf": {"enabled": False},
                "delta_stft": {"enabled": False},
                "cyc_doaemb":{
                    "enabled": True,
                    "cyc_alpha": 20,
                    "cyc_dimension": 40,
                    "use_ele": True,
                    "out_channel": 96, 
                    "fusion": "multiply" # concat or multiply
                }
            }
        }
        self.spatial_configs = deep_update(spatial_configs, config.get('spatial', {}))
        
        # --- 3. Dynamic Input Size Calculation ---
        spec_feat_dim = 2 
        
        n_pairs = len(self.spatial_configs['pairs'])
        feat_cfg = self.spatial_configs['features']
        spatial_dim=0
        if feat_cfg.get('ipd', {}).get('enabled', False): spatial_dim += n_pairs
        if feat_cfg.get('cdf', {}).get('enabled', False): spatial_dim += n_pairs
        if feat_cfg.get('sdf', {}).get('enabled', False): spatial_dim += n_pairs
        if feat_cfg.get('delta_stft', {}).get('enabled', False): spatial_dim += 2*n_pairs
        if feat_cfg.get('cyc_doaemb',{}).get('enabled',False): 
            if feat_cfg['cyc_doaemb']['fusion'] == "concat":
                spatial_dim += feat_cfg['cyc_doaemb']['out_channel']
            elif feat_cfg['cyc_doaemb']['fusion'] == "multiply":
                spatial_dim += 0
        
        total_input_size = spec_feat_dim + spatial_dim
        # print(f"Dynamic Input Size: {total_input_size}") # Debug用

        # --- 4. Backbone Configs ---
        block_kwargs = {
            'n_heads': 2,
            'dropout': 0.1,
            'conv_kernel_size': 3,
            'n_conv_groups': 8,
            'norms': ("LN", "GBN", "GBN"),
            'group_batch_norm_kwargs': {
                'group_size': freq_bins,
                'share_along_sequence_dim': False,
            },
        }
        
        sep_configs = dict(
            input_size=total_input_size, # 使用动态计算的值
            output_size=2, # 假设 NBC2 内部处理这里代表输出 mask 或 complex
            n_layers=8,
            dim_hidden=96,
            dim_ffn=96*2,
            block_kwargs=block_kwargs
        )
        self.sep_configs = deep_update(sep_configs, config.get('separator', {}))
        
        # --- 5. Instantiate Modules ---
        self.sep_model = NBC2(**self.sep_configs)
        
        self.n_layers = self.sep_configs['n_layers']
        self.clue_encoders = nn.ModuleList([
            CycEncoder(self.spatial_configs['features']['cyc_doaemb']) 
            for _ in range(self.n_layers)
        ])
        
    def forward(self, mix,cue):
        # input shape: (B, C, T)
        # self.window 已经在正确的 device 上了
        spatial_cue=cue[0]      
        B, M, T_wav = mix.shape
        self.window = self.window.to(mix.device)
        # print(f"mix_shape:{mix.shape}")
        # print(f"azimuth:{azi_rad},elerad:{ele_rad}")
        mix_reshape = mix.view(B * M, T_wav)
        
        spec = torch.stft(
            mix_reshape,
            n_fft=self.win,
            hop_length=self.stride,
            window=self.window,
            return_complex=True
        )
        
        _, F_dim, T_dim = spec.shape
        Y = spec.view(B, M, F_dim, T_dim)
        
        # --- 2. A-Norm ---
        Y_ref = Y[:, 0]
        ref_mag_mean = torch.abs(Y_ref).mean(dim=(1, 2), keepdim=True) + 1e-8
        Y_norm = Y / ref_mag_mean.unsqueeze(1)
        
        # Spectral: (B, 2, F, T)
        spec_feat = torch.stack([Y_norm[:, 0].real, Y_norm[:, 0].imag], dim=1)
        
        x = self.sep_model.encoder(spec_feat)
        
        # 2. 迭代式交互 (Loop with Interaction)
        # 同时遍历每一层的 Block 和 每一层的 Clue Encoder
        for block, clue_enc in zip(self.sep_model.sa_layers, self.clue_encoders):
            
            # A. 计算当前层的 DOA Embedding
            # Output: (B, Hidden, 1, T)
            doa_emb = clue_enc.compute(azi=spatial_cue[:, 0], ele=spatial_cue[:, 1])
            
            # B. 执行融合 (DSENet: Element-wise Multiply)
            # 这里利用广播机制: (B, H, F, T) * (B, H, 1, T) -> (B, H, F, T)
            # 无需任何手动 reshape/view，非常清爽
            if doa_emb is not None:
                x = clue_enc.post(x,doa_emb)
            
            # C. 通过当前层 Block
            # Input: (B, H, F, T) -> Output: (B, H, F, T)
            x, _ = block(x)
            
        # 3. 解码 (Decoder)
        # Input: (B, Hidden, F, T) -> Output: (B, Out, F, T)
        est_spec_feat = self.sep_model.decoder(x)
        # --- Reconstruction ---
        est_spec = torch.complex(est_spec_feat[:, 0], est_spec_feat[:, 1])
        
        # Inverse Normalization
        est_spec = est_spec * ref_mag_mean
        
        # --- iSTFT ---
        est_wav = torch.istft(
            est_spec,
            n_fft=self.win,
            hop_length=self.stride,
            window=self.window,
            length=T_wav
        )
        
        est_wav=est_wav.unsqueeze(1) # [B 1 T]
        
        return est_wav