import torch
import torch.nn as nn
import torch.nn.functional as F

from wesep.modules.spatial.spatial_frontend import SpatialFrontend
from wesep.modules.separator.bsrnn import BSRNN
from wesep.modules.common.deep_update import deep_update

class TSE_BSRNN_SPATIAL(nn.Module):
    def __init__(self, **config):
        super().__init__()
        
        # --- 1. Basic Configs ---
        self.win = config.get("win", 512)
        self.stride = config.get("stride", 256)
        
        self.register_buffer("window", torch.hann_window(self.win))
        
        # --- 2. Spatial Configs ---
        spatial_configs = {
            "geometry": {
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
                "ipd": {"enabled": True},
                "cdf": {"enabled": True},
                "sdf": {"enabled": True},
                "delta_stft": {"enabled": True},
            }
        }
        self.spatial_configs = deep_update(spatial_configs, config.get('spatial', {}))
        
        spec_feat_dim = 2 
        
        n_pairs = len(self.spatial_configs['pairs'])
        feat_cfg = self.spatial_configs['features']
        spatial_dim = 0
        if feat_cfg.get('ipd', {}).get('enabled', False): spatial_dim += n_pairs
        if feat_cfg.get('cdf', {}).get('enabled', False): spatial_dim += n_pairs
        if feat_cfg.get('sdf', {}).get('enabled', False): spatial_dim += n_pairs
        if feat_cfg.get('delta_stft', {}).get('enabled', False): spatial_dim += 2 * n_pairs
        
        self.total_input_size = spec_feat_dim + spatial_dim
        # print(f"Dynamic Input Size: {self.total_input_size}") 

        # --- 4. Backbone Configs ---
        sep_configs = dict(
            sr=16000,
            win=512,
            stride=128,
            feature_dim=128,
            num_repeat=6,
            causal=False,
            nspk=1,
            spec_dim=self.total_input_size, 
        )
        user_sep_cfg = config.get('separator', {})
        user_sep_cfg['spec_dim'] = self.total_input_size
        
        self.sep_configs = deep_update(sep_configs, user_sep_cfg)
        
        self.sep_model = BSRNN(**self.sep_configs)
        self.spatial_ft = SpatialFrontend(self.spatial_configs)
        
    def forward(self, mix, azi_rad, ele_rad=None):
        """
        mix: (B, M, T)
        """
        B, M, T_wav = mix.shape
        
        # --- 1. STFT ---
        mix_reshape = mix.view(B * M, T_wav)
        spec = torch.stft(
            mix_reshape,
            n_fft=self.win,
            hop_length=self.stride,
            window=self.window,
            return_complex=True
        )
        _, F_dim, T_dim = spec.shape
        Y = spec.view(B, M, F_dim, T_dim) # (B, M, F, T) Complex
        
        # --- 2. Reference & Norm ---
        Y_ref = Y[:, 0] # (B, F, T)
        ref_mag_mean = torch.abs(Y_ref).mean(dim=(1, 2), keepdim=True) + 1e-8
        Y_norm = Y / ref_mag_mean.unsqueeze(1)
        
        # --- 3. Feature Extraction ---
        # Spectral: (B, 2, F, T) -> Real, Imag
        spec_feat = torch.stack([Y_norm[:, 0].real, Y_norm[:, 0].imag], dim=1)
        
        # Spatial: (B, n_spatial, F, T)
        spatial_feat_dict = self.spatial_ft.compute_all(Y_norm,azi_rad, ele_rad)
        spatial_feat_list = []
        for name, feat in spatial_feat_dict.items():
            if name in self.spatial_configs["features"]:
                spatial_feat_list.append(feat)
        if len(spatial_feat_list) == 0:
            raise RuntimeError("No spatial features enabled or computed!")
        
        spatial_feat = torch.cat(spatial_feat_list, dim=1)
        
        # Fusion: (B, Total_Feats, F, T)
        features = torch.cat([spec_feat, spatial_feat], dim=1)
        
        subband_features = self.sep_model.band_split(features)
        
        # 4.2 Band Split (Reference Complex Spec)
        subband_mix_spec = self.sep_model.band_split(Y[:, 0]) 
        
        subband_feat_proj = self.sep_model.subband_norm(subband_features)
        
        # 4.4 Separation (BSRNN Backbone)
        sep_output = self.sep_model.separator(subband_feat_proj)
        
        # 4.5 Band Masker
        est_spec_RI = self.sep_model.band_masker(sep_output, subband_mix_spec)
        
        # --- 5. Reconstruction ---
        est_complex = torch.complex(est_spec_RI[:, 0], est_spec_RI[:, 1]) # (B, nspk, F, T)
        
        est_complex = est_complex * ref_mag_mean.unsqueeze(1)
        
        if self.sep_configs['nspk'] == 1:
            est_complex = est_complex.squeeze(1)
            
        est_wav = torch.istft(
            est_complex,
            n_fft=self.win,
            hop_length=self.stride,
            window=self.window,
            length=T_wav
        )
        
        return est_wav