import torch
import torch.nn as nn
import numpy as np
from wesep.modules.spatial.spatial_frontend import SpatialFrontend
from wesep.modules.separator.bsrnn import (
    BandSplit, 
    SubbandNorm, 
    BSRNN_Separator, 
    BandMasker
)
from wesep.modules.common.deep_update import deep_update

class TSE_BSRNN_SPATIAL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sr = 16000
        self.win = config.get("win", 512)
        self.stride = config.get("stride", 256)
        self.register_buffer("window", torch.hann_window(self.win))
        
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
                "cyc_doaemb":{
                    "enabled": True,
                    "cyc_alpha": 20,
                    "cyc_dimension": 40,
                    "use_ele": True,
                    "out_channel": 1, # only use when concat
                    "fusion_type": "multiply" # concat or multiply
                }
            }
        }
        self.spatial_configs = deep_update(spatial_configs, config.get('spatial', {}))
        self.spatial_ft = SpatialFrontend(self.spatial_configs)
        
        self.pairs = self.spatial_configs['pairs']
        self.mic_pos = torch.tensor(self.spatial_configs['geometry']['mic_coords'])
        self.c = self.spatial_configs['geometry']['c']
        self.fs = self.spatial_configs['geometry']['fs']
        
        sep_cfg = config.get('separator', {})
        feature_dim = sep_cfg.get('feature_dim', 128)
        num_repeat = sep_cfg.get('num_repeat', 6)
        causal = sep_cfg.get('causal', False)
        norm_type = "cLN" if causal else "GN"
        
        enc_dim = self.win // 2 + 1
        bandwidth_100 = int(np.floor(100 / (self.sr / 2.0) * enc_dim))
        bandwidth_200 = int(np.floor(200 / (self.sr / 2.0) * enc_dim))
        bandwidth_500 = int(np.floor(500 / (self.sr / 2.0) * enc_dim))
        band_width = [bandwidth_100]*10 + [bandwidth_200]*10 + [bandwidth_500]*5
        band_width.append(enc_dim - sum(band_width)) 
        
        self.nband = len(band_width)
        self.band_split = BandSplit(band_width)

        n_pairs = len(self.pairs)

        feat_cfg = self.spatial_configs['features']
        self.bandnorm = nn.ModuleDict()
        self.bandnorm['spec'] = SubbandNorm(
            band_width=band_width,
            spec_dim=n_pairs,
            nband=self.nband,
            feature_dim=feature_dim,
            norm_type=norm_type
        )
        
        if feat_cfg.get('ipd', {}).get('enabled', False): 
            self.bandnorm['ipd'] = SubbandNorm(
                band_width=band_width,
                spec_dim=n_pairs,
                nband=self.nband,
                feature_dim=feature_dim,
                norm_type=norm_type
            )
        if feat_cfg.get('cdf', {}).get('enabled', False): 
            self.bandnorm['cdf'] = SubbandNorm(
                band_width=band_width,
                spec_dim=n_pairs,
                nband=self.nband,
                feature_dim=feature_dim,
                norm_type=norm_type
            )
        if feat_cfg.get('sdf', {}).get('enabled', False): 
            self.bandnorm['sdf'] = SubbandNorm(
                band_width=band_width,
                spec_dim=n_pairs,
                nband=self.nband,
                feature_dim=feature_dim,
                norm_type=norm_type
            )
        if feat_cfg.get('delta_stft', {}).get('enabled', False): 
            self.bandnorm['delta_stft'] = SubbandNorm(
                band_width=band_width,
                spec_dim=2 * n_pairs,
                nband=self.nband,
                feature_dim=feature_dim,
                norm_type=norm_type
            )
        if feat_cfg.get('cyc_doaemb',{}).get('enabled',False): 
            if feat_cfg.get('cyc_doaemb',{}).get('fusion_type') == "concat":
                self.bandnorm['cyc_doaemb'] = SubbandNorm(
                    band_width=band_width,
                    spec_dim=n_pairs,
                    nband=feat_cfg['cyc_doaemb']['out_channel'],
                    feature_dim=feature_dim,
                    norm_type=norm_type
                )
            elif feat_cfg.get('cyc_doaemb',{}).get('fusion_type') == "multiply":
                feat_cfg['cyc_doaemb']['out_channel'] = 96 # dim_hidden    
        
        self.separator = BSRNN_Separator(
            nband=self.nband,
            num_repeat=num_repeat,
            feature_dim=feature_dim,
            norm_type=norm_type,
            causal=causal
        )
        
        self.band_masker = BandMasker(
            band_width=band_width,
            nband=self.nband,
            feature_dim=feature_dim,
            norm_type=norm_type,
            nspk=1
        )

    def forward(self, mix, cue):
        """
        mix: (B, M, T_wav)
        cue: (B, 2) -> [azimuth, elevation] (弧度)
        """
        B, M, T_wav = mix.shape
        device = mix.device
        spatial_cue = cue[0]
        azi_rad = spatial_cue[:, 0]
        ele_rad = spatial_cue[:, 1] 

        # --- 1. STFT ---
        mix_reshape = mix.view(B * M, T_wav)
        spec = torch.stft(
            mix_reshape, n_fft=self.win, hop_length=self.stride, 
            window=self.window.to(device), return_complex=True
        )
        _, F_dim, T_dim = spec.shape
        Y = spec.view(B, M, F_dim, T_dim)
        
        Y_ref = Y[:, 0]
        ref_mag_mean = torch.abs(Y_ref).mean(dim=(1, 2), keepdim=True) + 1e-8
        Y_norm = Y / ref_mag_mean.unsqueeze(1)

        spec_feat = torch.stack([Y_norm[:, 0].real, Y_norm[:, 0].imag], dim=1) 
        
        spatial_feat_dict = self.spatial_ft.compute_all(Y_norm, azi_rad, ele_rad)
        
        
        sub_spec = self.band_split(spec_feat)
        emb_spec = self.bandnorm['spec'](sub_spec)
        input_emb = emb_spec
        
        if self.spatial_configs['features']['ipd']['enabled'] :
            ipd_feat = spatial_feat_dict['ipd']
            sub_ipd = self.band_split(ipd_feat)
            emb_ipd = self.bandnorm['ipd'](sub_ipd)
            input_emb += emb_ipd
        
        if self.spatial_configs['features']['cdf']['enabled'] :
            cdf_feat = spatial_feat_dict['cdf']
            sub_cdf = self.band_split(cdf_feat)
            emb_cdf = self.bandnorm['cdf'](sub_cdf)
            input_emb += emb_cdf
        
        if self.spatial_configs['features']['sdf']['enabled']:
            sdf_feat = spatial_feat_dict['sdf']
            sub_sdf = self.band_split(sdf_feat)
            emb_sdf = self.bandnorm['sdf'](sub_sdf)
            input_emb += emb_sdf
        
        if self.spatial_configs['features']['delta_stft']['enabled']:
            stft_feat = spatial_feat_dict['delta_stft']
            sub_stft = self.band_split(stft_feat)
            emb_stft = self.bandnorm['delta_stft'](sub_stft)
            input_emb += emb_stft
        
        if self.spatial_configs['features']['cyc_doaemb']['enabled'] and self.spatial_configs['features']['cyc_doaemb']['fusion_type'] == 'concat':
            emb_feat = spatial_feat_dict['cyc_doaemb']
            sub_emb = self.band_split(emb_feat)
            emb_emb = self.bandnorm['cyc_doaemb'](sub_emb)
            input_emb += emb_emb
        
        if self.spatial_configs['features']['cyc_doaemb']['enabled'] and self.spatial_configs['features']['cyc_doaemb']['fusion_type'] == 'multiply':
            input_emb=self.spatial_ft.features['cyc_doaemb'].post(input_emb,spatial_feat_dict['cyc_doaemb'])     
        
        sep_out = self.separator(input_emb)   
        
        subband_mix = self.band_split(Y[:, 0])
        est_spec_RI = self.band_masker(sep_out, subband_mix)
        
        est_complex = torch.complex(est_spec_RI[:, 0], est_spec_RI[:, 1])
        est_complex = est_complex * ref_mag_mean.unsqueeze(1)
        
        est_wav = torch.istft(
            est_complex.squeeze(1), n_fft=self.win, hop_length=self.stride, 
            window=self.window.to(device), length=T_wav
        )
        
        return est_wav.unsqueeze(1)