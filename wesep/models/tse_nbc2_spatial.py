import torch
import torch.nn as nn
import torch.nn.functional as F

from wesep.modules.spatial.spatial_frontend import SpatialFrontend
from wesep.modules.separator.nbc2 import NBC2
from wesep.modules.common.deep_update import deep_update

class TSE_NBC2_SPATIAL(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # --- 1. top model setting ---
        self.full_input = config.get("full_input",True)
        
        # --- 2. Merge Configs ---
        spatial_configs = {
            "geometry": {
                "n_fft": 512,              
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
                "cyc_doaemb":{
                    "encoder_kwargs" :{
                        "cyc_alpha": 20,
                        "cyc_dimension": 40,
                        "use_ele": True,
                        "out_channel": 1,
                    },
                    "enabled": True,
                    "num_encoder" : 1 , # use when multiply 
                }
            }
        }
        self.spatial_configs = deep_update(spatial_configs, config.get('spatial', {}))
        block_kwargs = {
            'n_heads': 2,
            'dropout': 0.1,
            'conv_kernel_size': 3,
            'n_conv_groups': 8,
            'norms': ("LN", "GBN", "GBN"),
            'group_batch_norm_kwargs': {
                'group_size': 257,
                'share_along_sequence_dim': False,
            },
        }
        sep_configs = dict(
            win=512,
            stride=256,
            input_size=2, # for only ref-channel input 
            output_size=2,
            n_layers=8,
            dim_hidden=96,
            dim_ffn=96*2,
            block_kwargs=block_kwargs,
        )
        self.sep_configs = deep_update(sep_configs, config.get('separator', {}))
        # --- 3. Dynamic Input Size Calculation ---
        ### spec_feat dim calculation
        n_pairs = len(self.spatial_configs['pairs'])
        if self.full_input :
            self.sep_configs["input_size"] = 2 * len(self.spatial_configs['geometry']['mic_coords'])
            
        feat_cfg = self.spatial_configs['features']
        if feat_cfg.get('ipd', {}).get('enabled', False): 
            self.sep_configs["input_size"] += n_pairs
        if feat_cfg.get('cdf', {}).get('enabled', False): 
            self.sep_configs["input_size"] += n_pairs
        if feat_cfg.get('sdf', {}).get('enabled', False): 
            self.sep_configs["input_size"] += n_pairs
        if feat_cfg.get('delta_stft', {}).get('enabled', False):
            self.sep_configs["input_size"] += 2*n_pairs
        if feat_cfg.get('cyc_doaemb',{}).get('enabled',False): 
            self.spatial_configs['features']['cyc_doaemb']['encoder_kwargs']['out_channel'] = self.sep_configs["dim_hidden"] # dim_hidden    
            self.spatial_configs['features']['cyc_doaemb']['num_encoder'] = self.sep_configs["n_layers"] 
        # --- 5. Instantiate Modules ---
        self.sep_model = NBC2(**self.sep_configs)
        self.spatial_ft = SpatialFrontend(self.spatial_configs)
        
    def forward(self, mix, cue):
        # input shape: (B, C, T)
        spatial_cue=cue[0]
        azi_rad = spatial_cue[:, 0]
        ele_rad = spatial_cue[:, 1] 
               
        # S1. Convert into frequency-domain
        spec = self.sep_model.stft(mix)[-1]
        
        # S2. Concat real and imag, split to subbands
        # Spectral: (B, 2, F, T) or (B, C, F, T)
        spec_feat = None
        if self.full_input:
            spec_feat = torch.cat([spec.real, spec.imag], dim=1)
        else :    
            spec_feat = torch.stack([spec[:, 0].real, spec[:, 0].imag], dim=1)
        
        # spatial_feat_dict = self.spatial_ft.compute_all(spec, azi_rad, ele_rad)
        #######################################################
        # Spatio-temporal Features
        # Spatial: (B, 16, F, T)
        if self.spatial_configs['features']['ipd']['enabled'] :
            ipd_feature = self.spatial_ft.features['ipd'].compute(spec)
            spec_feat = self.spatial_ft.features['ipd'].post(spec_feat,ipd_feature)
            # spec_feat=self.spatial_ft.features['ipd'].post(spec_feat,spatial_feat_dict['ipd'])
        
        if self.spatial_configs['features']['cdf']['enabled'] :
            cdf_feature = self.spatial_ft.features['cdf'].compute(spec,azi_rad,ele_rad)
            spec_feat = self.spatial_ft.features['ipd'].post(spec_feat,cdf_feature)
            # spec_feat=self.spatial_ft.features['cdf'].post(spec_feat,spatial_feat_dict['cdf'])
        
        if self.spatial_configs['features']['sdf']['enabled']:
            sdf_feature = self.spatial_ft.features['sdf'].compute(spec,azi_rad,ele_rad)
            spec_feat = self.spatial_ft.features['ipd'].post(spec_feat,sdf_feature)
            # spec_feat=self.spatial_ft.features['sdf'].post(spec_feat,spatial_feat_dict['sdf'])
        
        if self.spatial_configs['features']['delta_stft']['enabled']:
            dstft_feature = self.spatial_ft.features['delta_stft'].compute(spec)
            spec_feat = self.spatial_ft.features['ipd'].post(spec_feat,dstft_feature)
            # spec_feat=self.spatial_ft.features['delta_stft'].post(spec_feat,spatial_feat_dict['delta_stft'])
            
        ####################################################
        encode_features = self.sep_model.encoder(spec_feat) # Conv
        for idx,m in enumerate(self.sep_model.sa_layers): # nbc2_block
            # cyc_doaemb ele-multiply
            if self.spatial_configs['features']['cyc_doaemb']['enabled']:
                cyc_doaemb = self.spatial_ft.features['cyc_doaemb'].compute(azi_rad,ele_rad,layer_idx=idx)
                encode_features=self.spatial_ft.features['cyc_doaemb'].post(encode_features,cyc_doaemb,layer_idx=idx)
            encode_features , _ = m(encode_features)
        
        est_spec_feat = self.sep_model.decoder(encode_features)
        
        est_spec = torch.complex(est_spec_feat[:, 0], est_spec_feat[:, 1])
        
        est_wav = self.sep_model.istft(est_spec)
        return est_wav
    