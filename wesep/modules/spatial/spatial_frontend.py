import torch
import torch.nn as nn
import math
from wesep.modules.common.deep_update import deep_update
from wesep.modules.feature.speech import STFT

class BaseSpatialFeature(nn.Module):
    def __init__(self, config, geometry_ctx):
        """
        geometry_ctx: 包含几何常量的字典或对象
        config: 可能包含 'pairs' 作为默认值
        """
        super().__init__()
        self.default_pairs = config.get('pairs', None)
        self.register_buffer('mic_pos', geometry_ctx['mic_pos'])
        self.register_buffer('omega_over_c', geometry_ctx['omega_over_c'])

    def _get_pairs(self, pairs_arg):
        if pairs_arg is not None:
            return pairs_arg
        if self.default_pairs is not None:
            return self.default_pairs
        raise ValueError(f"{self.__class__.__name__}: No pairs provided in arg or config.")

    def _compute_tpd(self, azi, ele, F_dim, pairs):
        u_x = torch.cos(ele) * torch.cos(azi)
        u_y = torch.cos(ele) * torch.sin(azi)
        u_z = torch.sin(ele)
        u_vec = torch.stack([u_x, u_y, u_z], dim=1) 

        d_vecs = []
        for (i, j) in pairs:
            d_vecs.append(self.mic_pos[i] - self.mic_pos[j])
        d_tensor = torch.stack(d_vecs, dim=0) 

        dist_delay = torch.matmul(u_vec, d_tensor.T) 

        TPD = self.omega_over_c.view(1, 1, F_dim, 1) * dist_delay.unsqueeze(-1).unsqueeze(-1)
        return TPD 

    def compute(self, Y, azi, ele, pairs=None):
        raise NotImplementedError

    def post(self, mix_repr, spatial_repr):
        raise NotImplementedError

class IPDFeature(BaseSpatialFeature):
    def compute(self, Y, azi, ele, pairs=None):
        target_pairs = self._get_pairs(pairs)
        ipd_list = []
        for (i, j) in target_pairs:
            diff = Y[:, i].angle() - Y[:, j].angle()
            diff = torch.remainder(diff + math.pi, 2 * math.pi) - math.pi
            ipd_list.append(diff)
        return torch.stack(ipd_list, dim=1) # (B, N, F, T)

    def post(self, mix_repr, spatial_repr):
        return torch.cat([mix_repr, spatial_repr], dim=1)


class CDFFeature(BaseSpatialFeature):
    def compute(self, Y, azi, ele, pairs=None):
        target_pairs = self._get_pairs(pairs)
        ipd_list = []
        for (i, j) in target_pairs:
            diff = Y[:, i].angle() - Y[:, j].angle()
            diff = torch.remainder(diff + math.pi, 2 * math.pi) - math.pi
            ipd_list.append(diff)
        IPD = torch.stack(ipd_list, dim=1)
        
        _, _, F_dim, _ = Y.shape
        TPD = self._compute_tpd(azi, ele, F_dim, target_pairs)
        
        return torch.cos(IPD - TPD)

    def post(self, mix_repr, spatial_repr):
        return torch.cat([mix_repr, spatial_repr], dim=1)


class SDFFeature(BaseSpatialFeature):
    def compute(self, Y, azi, ele, pairs=None):
        target_pairs = self._get_pairs(pairs)
        ipd_list = []
        for (i, j) in target_pairs:
            diff = Y[:, i].angle() - Y[:, j].angle()
            diff = torch.remainder(diff + math.pi, 2 * math.pi) - math.pi
            ipd_list.append(diff)
        IPD = torch.stack(ipd_list, dim=1)
        
        _, _, F_dim, _ = Y.shape
        TPD = self._compute_tpd(azi, ele, F_dim, target_pairs)
        
        return torch.sin(IPD - TPD)

    def post(self, mix_repr, spatial_repr):
        return torch.cat([mix_repr, spatial_repr], dim=1)

class DSTFTFeature(BaseSpatialFeature):
    def compute(self, Y, azi, ele, pairs=None):
        target_pairs = self._get_pairs(pairs)
        d_list = []
        
        for (i, j) in target_pairs:
            diff = Y[:, i] - Y[:, j]
        
            d_list.append(diff.real)
            d_list.append(diff.imag)

        return torch.stack(d_list, dim=1)

    def post(self, mix_repr, spatial_repr):
        return torch.cat([mix_repr, spatial_repr], dim=1)


class SpatialFrontend(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # ===== Default Config =====
        DEFAULT_CONFIG = {
            "geometry": {
                "n_fft": 512,
                "hop_length": 128,
                "win_length": 512,
                "fs": 16000,
                "c": 343.0,
                "mic_spacing": 0.033333,
                "mic_coords": [
                    [-0.05,        0.0, 0.0],  # Mic 0
                    [-0.01666667,  0.0, 0.0],  # Mic 1
                    [ 0.01666667,  0.0, 0.0],  # Mic 2
                    [ 0.05,        0.0, 0.0],  # Mic 3
                ],
            },
            "pairs": [[0, 1], [1, 2], [2, 3], [0, 3]], 
            "features": {
                "ipd": {"enabled": True},
                "cdf": {"enabled": True},
                "sdf": {"enabled": True},
                "delta_stft": {"enabled": True},
            }
        }
        self.config = deep_update(DEFAULT_CONFIG, config)
        geo_cfg = self.config['geometry']
        
        self.stft = STFT(
            n_fft=geo_cfg['n_fft'], 
            hop_length=geo_cfg['hop_length'], 
            win_length=geo_cfg['win_length']
        )
        
        freq_bins = geo_cfg['n_fft'] // 2 + 1
        freq_vec = torch.linspace(0, geo_cfg['fs'] / 2, freq_bins)
        
        if 'mic_coords' in geo_cfg:
            mic_pos = torch.tensor(geo_cfg['mic_coords'])
        else:
            M = 4 
            spacing = geo_cfg['mic_spacing']
            mic_pos = torch.zeros(M, 3)
            mic_pos[:, 0] = torch.arange(M) * spacing

        geometry_ctx = {
            'mic_pos': mic_pos,
            'omega_over_c': 2 * math.pi * freq_vec / geo_cfg['c']
        }
        
        self.features = nn.ModuleDict()
        self.default_pairs = self.config['pairs']
        feat_cfg = self.config['features']
        
        if feat_cfg['ipd']['enabled']:
            self.features['ipd'] = IPDFeature({'pairs': self.default_pairs}, geometry_ctx)
        
        if feat_cfg['cdf']['enabled']:
            self.features['cdf'] = CDFFeature({'pairs': self.default_pairs}, geometry_ctx)

        if feat_cfg['sdf']['enabled']:
            self.features['sdf'] = SDFFeature({'pairs': self.default_pairs}, geometry_ctx)

        if feat_cfg['delta_stft']['enabled']:
            self.features['delta_stft'] = DSTFTFeature({'pairs': self.default_pairs}, geometry_ctx)

    def compute_all(self, Y, azi, ele=None, pairs=None):
        """
        计算所有启用特征
        """
        if ele is None:
            ele = torch.zeros_like(azi)
        
        out = {}
        for name, module in self.features.items():
            out[name] = module.compute(Y, azi, ele, pairs=pairs)
            
        return out
