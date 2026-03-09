import torch
import torch.nn as nn
import math
import copy
from wesep.modules.common.deep_update import deep_update
from wesep.modules.spatial.pos_encoding import PosEncodingFactory

class BaseSpatialFeature(nn.Module):
    def __init__(self, config, geometry_ctx=None):
        super().__init__()
        self.config=config
        self.default_pairs = config.get('pairs', None)
        if geometry_ctx is not None:
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

    def compute(self, azi=None, ele=None, Y=None,pairs=None):
        raise NotImplementedError

    def post(self, mix_repr, spatial_repr):
        raise NotImplementedError
class SpatialEncoderGroup(nn.Module):
    def __init__(self, base_encoder: nn.Module, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([copy.deepcopy(base_encoder) for _ in range(num_layers)])

    def compute(self, azi=None, ele=None, Y=None, pairs=None, layer_idx=None):
        if layer_idx is not None:
            return self.layers[layer_idx].compute(azi=azi, ele=ele, Y=Y, pairs=pairs)
            
        results = []
        for layer in self.layers:
            results.append(layer.compute(azi=azi, ele=ele, Y=Y, pairs=pairs))
        return results

    def post(self, mix_repr, spatial_repr, layer_idx=None):
        if layer_idx is not None:
            return self.layers[layer_idx].post(mix_repr, spatial_repr)
            
        if self.num_layers == 1:
            repr_item = spatial_repr[0] if isinstance(spatial_repr, list) else spatial_repr
            return self.layers[0].post(mix_repr, repr_item)
            
        out_repr = mix_repr
        for i, layer in enumerate(self.layers):
            out_repr = layer.post(out_repr, spatial_repr[i])
            
        return out_repr
class InitStatesFeature(BaseSpatialFeature): 
    def __init__(self, config, geometry_ctx=None):
        super().__init__(config, geometry_ctx)
        
        self.hidden_size_f = config["hidden_size_f"]
        self.hidden_size_t = config["hidden_size_t"]
        self.use_ele = config.get("use_ele", False)
        
        encoding_cfg = config.get("encoding_config", {})
        self.encoder, self.enc_dim = PosEncodingFactory.create(encoding_cfg, self.use_ele)
        
        self.encoding_type = encoding_cfg.get("encoding", "oh")
        
        self.proj_band_h0 = nn.Linear(self.enc_dim, self.hidden_size_f)
        self.proj_band_c0 = nn.Linear(self.enc_dim, self.hidden_size_f)
        self.proj_comm_h0 = nn.Linear(self.enc_dim, self.hidden_size_t)
        self.proj_comm_c0 = nn.Linear(self.enc_dim, self.hidden_size_t)

    def compute(self, azi, ele=None, Y=None, pairs=None):
        is_missing = (azi <= -998.0)
        
        safe_azi = torch.where(is_missing, torch.zeros_like(azi), azi)
        
        if is_missing.dim() == 2:
            is_missing = is_missing[:, 0]
        is_missing = is_missing.view(-1, 1)

        if safe_azi.dim() == 2: safe_azi = safe_azi[:, 0]
        
        if ele is not None:
            safe_ele = torch.where(ele <= -998.0, torch.zeros_like(ele), ele)
            if safe_ele.dim() == 2: safe_ele = safe_ele[:, 0]
        else:
            safe_ele = None
        doa_enc = self.encoder(safe_azi)
        if self.use_ele and safe_ele is not None:
            ele_input = torch.abs(safe_ele) if self.encoding_type in ["oh", "onehot"] else safe_ele
            ele_enc = self.encoder(ele_input)
            doa_enc = torch.cat([doa_enc, ele_enc], dim=-1)
        band_h0 = self.proj_band_h0(doa_enc)
        band_c0 = self.proj_band_c0(doa_enc)
        comm_h0 = self.proj_comm_h0(doa_enc)
        comm_c0 = self.proj_comm_c0(doa_enc)
        zeros_f = torch.zeros_like(band_h0)
        zeros_t = torch.zeros_like(comm_h0)
        
        return {
            "band_h0": torch.where(is_missing, zeros_f, band_h0),
            "band_c0": torch.where(is_missing, zeros_f, band_c0),
            "comm_h0": torch.where(is_missing, zeros_t, comm_h0),
            "comm_c0": torch.where(is_missing, zeros_t, comm_c0)
        }

    def post(self, mix_repr, spatial_repr):
        return mix_repr
        
class TimeVariantMultiplyFeature(BaseSpatialFeature): 
    def __init__(self, config, geometry_ctx=None):
        super().__init__(config, geometry_ctx)
        
        self.out_channels = config['out_channel']
        self.use_ele = config.get('use_ele', False)
        
        encoding_cfg = config.get("encoding_config", {})
        self.encoder, self.enc_dim = PosEncodingFactory.create(encoding_cfg, self.use_ele)
        self.encoding_type = encoding_cfg.get("encoding", "cyc")
        
        self.mlp = nn.Sequential(
            nn.Linear(self.enc_dim, self.out_channels),
            nn.LayerNorm(self.out_channels),
            nn.PReLU()
        )
    def compute(self, azi, ele=None, Y=None, pairs=None):
        if azi.dim() == 1: azi = azi.unsqueeze(1)
        if ele is not None and ele.dim() == 1: ele = ele.unsqueeze(1)
        
        doa_enc = self.encoder(azi)
        if self.use_ele and ele is not None:
            ele_input = torch.abs(ele) if self.encoding_type in ["oh", "onehot"] else ele
            ele_enc = self.encoder(ele_input)
            doa_enc = torch.cat([doa_enc, ele_enc], dim=-1)

        # Input: (B, T, enc_dim) -> Output: (B, T, out_channels)
        spatial_repr = self.mlp(doa_enc)
        spatial_repr = spatial_repr.permute(0, 2, 1).unsqueeze(2) # (B, C, 1, T)

        is_missing = (azi <= -998.0).view(-1, 1, 1, 1)
        return torch.where(is_missing, torch.ones_like(spatial_repr), spatial_repr)

    def post(self, mix_repr, spatial_repr):
        if spatial_repr is None:
            return mix_repr
        if mix_repr.shape[1] != spatial_repr.shape[1]:
            raise ValueError(
                f"Fusion 'multiply' requires same channel dimensions. "
                f"Mix: {mix_repr.shape[1]}, Spatial: {spatial_repr.shape[1]}."
            )
        return mix_repr * spatial_repr
    
class IPDFeature(BaseSpatialFeature):
    def compute(self, Y, azi=None, ele=None, pairs=None):
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
    def compute(self, Y, azi, ele=None, pairs=None):
        target_pairs = self._get_pairs(pairs)
        ipd_list = []
        for (i, j) in target_pairs:
            diff = Y[:, i].angle() - Y[:, j].angle()
            diff = torch.remainder(diff + math.pi, 2 * math.pi) - math.pi
            ipd_list.append(diff)
        IPD = torch.stack(ipd_list, dim=1)
        
        _, _, F_dim, _ = Y.shape
        TPD = self._compute_tpd(azi, ele, F_dim, target_pairs)
        
        out = torch.cos(IPD - TPD)
        
        is_missing = (azi <= -998.0).view(-1, 1, 1, 1) 
        return torch.where(is_missing, torch.ones_like(out), out)

    def post(self, mix_repr, spatial_repr):
        return torch.cat([mix_repr, spatial_repr], dim=1)


class SDFFeature(BaseSpatialFeature):
    def compute(self, Y, azi, ele=None, pairs=None):
        target_pairs = self._get_pairs(pairs)
        ipd_list = []
        for (i, j) in target_pairs:
            diff = Y[:, i].angle() - Y[:, j].angle()
            diff = torch.remainder(diff + math.pi, 2 * math.pi) - math.pi
            ipd_list.append(diff)
        IPD = torch.stack(ipd_list, dim=1)
        
        _, _, F_dim, _ = Y.shape
        TPD = self._compute_tpd(azi, ele, F_dim, target_pairs)
        
        out = torch.sin(IPD - TPD)
        
        is_missing = (azi <= -998.0).view(-1, 1, 1, 1)
        return torch.where(is_missing, torch.zeros_like(out), out)

    def post(self, mix_repr, spatial_repr):
        return torch.cat([mix_repr, spatial_repr], dim=1)

class DSTFTFeature(BaseSpatialFeature):
    def compute(self, Y, azi=None, ele=None, pairs=None):
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
                "ipd": {
                    "enabled": False, 
                    "num_encoder": 1
                },
                "cdf": {
                    "enabled": False, 
                    "num_encoder": 1
                },
                "sdf": {
                    "enabled": False, 
                    "num_encoder": 1
                },
                "delta_stft": {
                    "enabled": False, 
                    "num_encoder": 1
                },
                "Multiply_emb": {
                    "enabled": False,
                    "num_encoder": 1,
                    "encoding_config":{
                        "encoding": "cyc",
                        "cyc_alpha": 20,
                        "cyc_dimension": 40,
                    },
                    "use_ele": True,
                    "out_channel": 1
                },
                "InitStates_emb": {  
                    "enabled": False,
                    "num_encoder": 1,
                    "encoding_config":{
                        "encoding": "oh",
                        "emb_dim": 180,
                    },
                    "hidden_size_f": 256,
                    "hidden_size_t": 256,
                    "use_ele" : True
                }
            }
        }
        self.config = deep_update(DEFAULT_CONFIG, config)
        geo_cfg = self.config['geometry']
        
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
        
        FEATURE_REGISTRY = {
            'ipd': IPDFeature,
            'cdf': CDFFeature,
            'sdf': SDFFeature,
            'delta_stft': DSTFTFeature,
            'Multiply_emb': TimeVariantMultiplyFeature,
            'InitStates_emb': InitStatesFeature
        }
        
        for feat_name, sub_cfg in feat_cfg.items():
            if not sub_cfg.get('enabled', False):
                continue
                
            if feat_name not in FEATURE_REGISTRY:
                raise ValueError(f"Unknown spatial feature in config: {feat_name}")
                
            num_encoder = sub_cfg.get('num_encoder', 1)
            
            sub_cfg_with_pairs = deep_update({'pairs': self.default_pairs}, sub_cfg)
            
            base_module = FEATURE_REGISTRY[feat_name](sub_cfg_with_pairs, geometry_ctx)
            
            self.features[feat_name] = SpatialEncoderGroup(base_module, num_encoder)
        
    def compute_all(self, Y, azi, ele=None, pairs=None):
        if ele is None:
            ele = torch.zeros_like(azi)
        
        out = {}
        for name, module in self.features.items():
            out[name] = module.compute(Y=Y, azi=azi, ele=ele, pairs=pairs)
            
        return out