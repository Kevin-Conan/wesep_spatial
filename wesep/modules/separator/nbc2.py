import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from torch.nn import Module, MultiheadAttention
from torch.nn.parameter import Parameter
from typing import Any, Dict, Optional, Tuple, Union
import math


class LayerNorm(nn.LayerNorm):
    def __init__(self, transpose: bool, **kwargs) -> None:
        super().__init__(**kwargs)
        self.transpose = transpose

    def forward(self, input: Tensor) -> Tensor:
        if self.transpose:
            input = input.transpose(-1, -2)
        o = super().forward(input)
        if self.transpose:
            o = o.transpose(-1, -2)
        return o

class BatchNorm1d(nn.Module):
    def __init__(self, transpose: bool, **kwargs) -> None:
        super().__init__()
        self.transpose = transpose
        self.bn = nn.BatchNorm1d(**kwargs)

    def forward(self, input: Tensor) -> Tensor:
        if self.transpose == False:
            input = input.transpose(-1, -2)
        o = self.bn.forward(input)
        if self.transpose == False:
            o = o.transpose(-1, -2)
        return o

class GroupNorm(nn.GroupNorm):
    def __init__(self, transpose: bool, **kwargs) -> None:
        super().__init__(**kwargs)
        self.transpose = transpose

    def forward(self, input: Tensor) -> Tensor:
        if self.transpose == False:
            input = input.transpose(-1, -2)
        o = super().forward(input)
        if self.transpose == False:
            o = o.transpose(-1, -2)
        return o

class GroupBatchNorm(Module):
    dim_hidden: int
    group_size: int
    eps: float
    affine: bool
    transpose: bool
    share_along_sequence_dim: bool

    def __init__(self, dim_hidden: int, group_size: int, share_along_sequence_dim: bool = False, transpose: bool = False, affine: bool = True, eps: float = 1e-5) -> None:
        super(GroupBatchNorm, self).__init__()
        self.dim_hidden = dim_hidden
        self.group_size = group_size
        self.eps = eps
        self.affine = affine
        self.transpose = transpose
        self.share_along_sequence_dim = share_along_sequence_dim
        if self.affine:
            if transpose:
                self.weight = Parameter(torch.empty([dim_hidden, 1]))
                self.bias = Parameter(torch.empty([dim_hidden, 1]))
            else:
                self.weight = Parameter(torch.empty([dim_hidden]))
                self.bias = Parameter(torch.empty([dim_hidden]))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        assert (input.shape[0] // self.group_size) * self.group_size, f'batch size {input.shape[0]} is not divisible by group size {self.group_size}'
        if self.transpose == False:
            B, T, H = input.shape
            input = input.reshape(B // self.group_size, self.group_size, T, H)
            if self.share_along_sequence_dim:
                var, mean = torch.var_mean(input, dim=(1, 2, 3), unbiased=False, keepdim=True)
            else:
                var, mean = torch.var_mean(input, dim=(1, 3), unbiased=False, keepdim=True)
            output = (input - mean) / torch.sqrt(var + self.eps)
            if self.affine:
                output = output * self.weight + self.bias
            output = output.reshape(B, T, H)
        else:
            B, H, T = input.shape
            input = input.reshape(B // self.group_size, self.group_size, H, T)
            if self.share_along_sequence_dim:
                var, mean = torch.var_mean(input, dim=(1, 2, 3), unbiased=False, keepdim=True)
            else:
                var, mean = torch.var_mean(input, dim=(1, 2), unbiased=False, keepdim=True)
            output = (input - mean) / torch.sqrt(var + self.eps)
            if self.affine:
                output = output * self.weight + self.bias
            output = output.reshape(B, H, T)
        return output

    def extra_repr(self) -> str:
        return '{dim_hidden}, {group_size}, share_along_sequence_dim={share_along_sequence_dim}, transpose={transpose}, eps={eps}, affine={affine}'.format(**self.__dict__)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from torch.nn import Module, MultiheadAttention
from torch.nn.parameter import Parameter
from typing import Any, Dict, Optional, Tuple, Union
import math

# --- 1. 保持原有的 Norm 类不变 (LayerNorm, BatchNorm1d, GroupNorm, GroupBatchNorm) ---
# (为了节省篇幅，这里假设你上面提供的 Norm 类代码已经包含在这里了)
# ... [Paste your Norm classes here] ...

# --- 2. 新增: 封装的 Encoder (处理输入重整) ---
class NBC2Encoder(nn.Module):
    def __init__(self, input_size: int, dim_hidden: int, kernel_size: int = 5):
        super().__init__()
        # 使用 Conv1d 作为 Encoder，卷积核沿时间轴滑动
        self.conv = nn.Conv1d(
            in_channels=input_size, 
            out_channels=dim_hidden, 
            kernel_size=kernel_size, 
            stride=1, 
            padding="same"
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Input:  (B, C_in, F, T)
        Output: (B, C_out, F, T)
        """
        B, C, F, T = x.shape
        
        # 1. Merge Batch and Freq: (B, C, F, T) -> (B*F, C, T)
        # 这一步是为了让 Conv1d 将每个频率点的时间序列视为独立的样本处理
        x = x.permute(0, 2, 1, 3).contiguous().view(B * F, C, T)
        
        # 2. Conv1d Encoding
        x = self.conv(x)  # (B*F, dim_hidden, T)
        
        # 3. Restore dimensions: (B*F, dim_hidden, T) -> (B, dim_hidden, F, T)
        # 注意输出通道数变为 dim_hidden
        C_out = x.shape[1]
        x = x.view(B, F, C_out, T).permute(0, 2, 1, 3).contiguous()
        
        return x

# --- 3. 新增: 封装的 Decoder (处理输出重整) ---
class NBC2Decoder(nn.Module):
    def __init__(self, dim_hidden: int, output_size: int):
        super().__init__()
        # 原代码中使用 Linear 进行解码
        self.linear = nn.Linear(in_features=dim_hidden, out_features=output_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Input:  (B, C_in, F, T)
        Output: (B, C_out, F, T)
        """
        B, C, F, T = x.shape
        
        # 1. Prepare for Linear: (B, C, F, T) -> (B, F, T, C)
        # Linear 作用于最后一个维度
        x = x.permute(0, 2, 3, 1).contiguous()
        
        # 2. Linear Projection
        x = self.linear(x) # (B, F, T, output_size)
        
        # 3. Restore: (B, F, T, output_size) -> (B, output_size, F, T)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        return x

# --- 4. 修改: NBC2Block (内部处理维度转换) ---
class NBC2Block(nn.Module):
    def __init__(self, dim_hidden: int, dim_ffn: int, n_heads: int, dropout: float = 0, conv_kernel_size: int = 3, n_conv_groups: int = 8, norms: Tuple[str, str, str] = ("LN", "GBN", "GBN"), group_batch_norm_kwargs: Dict[str, Any] = {'group_size': 257, 'share_along_sequence_dim': False}) -> None:
        super().__init__()
        self.norm1 = self._new_norm(norms[0], dim_hidden, False, n_conv_groups, **group_batch_norm_kwargs)
        self.self_attn = MultiheadAttention(embed_dim=dim_hidden, num_heads=n_heads, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = self._new_norm(norms[1], dim_hidden, False, n_conv_groups, **group_batch_norm_kwargs)
        self.linear1 = nn.Linear(dim_hidden, dim_ffn)
        
        # FFN 中的 Conv 模块
        self.conv = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_ffn, kernel_size=conv_kernel_size, padding='same', groups=n_conv_groups, bias=True),
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_ffn, kernel_size=conv_kernel_size, padding='same', groups=n_conv_groups, bias=True),
            self._new_norm(norms[2], dim_ffn, True, n_conv_groups, **group_batch_norm_kwargs),
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_ffn, kernel_size=conv_kernel_size, padding='same', groups=n_conv_groups, bias=True),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.linear2 = nn.Linear(dim_ffn, dim_hidden)
        self.dropout2 = nn.Dropout(dropout)
        
        # Init weights
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: Tensor, att_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Input:  (B, C, F, T)  <-- 标准化接口
        Output: (B, C, F, T), attn
        """
        B, C, F, T = x.shape
        
        # --- 1. Reshape to internal format (Batch-First Sequence) ---
        # (B, C, F, T) -> (B, F, T, C) -> (B*F, T, C)
        # MultiheadAttention 和 Linear 需要 (Batch, Seq, Feature)
        x_internal = x.permute(0, 2, 3, 1).contiguous().view(B * F, T, C)

        # --- 2. Original Logic ---
        x_, attn = self._sa_block(self.norm1(x_internal), att_mask)
        x_internal = x_internal + x_
        x_internal = x_internal + self._ff_block(self.norm2(x_internal))
        
        # --- 3. Restore Shape ---
        # (B*F, T, C) -> (B, F, T, C) -> (B, C, F, T)
        x_out = x_internal.view(B, F, T, C).permute(0, 3, 1, 2).contiguous()
        
        return x_out, attn

    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        if isinstance(self.self_attn, MultiheadAttention):
            x, attn = self.self_attn.forward(x, x, x, average_attn_weights=False, attn_mask=attn_mask)
        else:
            x, attn = self.self_attn(x, attn_mask=attn_mask)
        return self.dropout1(x), attn

    def _ff_block(self, x: Tensor) -> Tensor:
        # Conv 需要 (Batch, Channel, Length)，所以这里有 transpose
        # x: (B*F, T, C) -> Linear -> (B*F, T, FFN_dim) -> Transpose -> (B*F, FFN_dim, T)
        x = self.linear2(self.conv(self.linear1(x).transpose(-1, -2)).transpose(-1, -2))
        return self.dropout2(x)

    def _new_norm(self, norm_type: str, dim_hidden: int, transpose: bool, num_conv_groups: int, **freq_norm_kwargs):
        # 保持原样
        if norm_type == 'LN':
            norm = LayerNorm(normalized_shape=dim_hidden, transpose=transpose)
        elif norm_type == 'GBN':
            norm = GroupBatchNorm(dim_hidden=dim_hidden, transpose=transpose, **freq_norm_kwargs)
        elif norm_type == 'BN':
            norm = BatchNorm1d(num_features=dim_hidden, transpose=transpose)
        elif norm_type == 'GN':
            norm = GroupNorm(num_groups=num_conv_groups, num_channels=dim_hidden, transpose=transpose)
        else:
            raise Exception(norm_type)
        return norm

# --- 5. 修改: NBC2 主模型 (Forward 极其清爽) ---
class NBC2(nn.Module):
    def __init__(self, input_size: int, output_size: int, n_layers: int, encoder_kernel_size: int = 5, dim_hidden: int = 192, dim_ffn: int = 384, block_kwargs: Dict[str, Any] = {}) -> None:
        super().__init__()
        
        # 1. 使用封装好的 Encoder
        self.encoder = NBC2Encoder(input_size=input_size, dim_hidden=dim_hidden, kernel_size=encoder_kernel_size)
        
        # 2. Block 列表
        self.sa_layers = nn.ModuleList()
        for l in range(n_layers):
            self.sa_layers.append(NBC2Block(dim_hidden=dim_hidden, dim_ffn=dim_ffn, **block_kwargs))
        
        # 3. 使用封装好的 Decoder
        self.decoder = NBC2Decoder(dim_hidden=dim_hidden, output_size=output_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Input:  (B, C_in, F, T)
        Output: (B, C_out, F, T)
        无需在主 forward 中进行任何 permute/view
        """
        # 1. Encoding: (B, In, F, T) -> (B, Hidden, F, T)
        x = self.encoder(x)
        
        # 2. Stacked Blocks: (B, Hidden, F, T) -> (B, Hidden, F, T)
        for m in self.sa_layers:
            # Block 内部自适应处理维度
            x, attn = m(x)
            del attn
            
        # 3. Decoding: (B, Hidden, F, T) -> (B, Out, F, T)
        y = self.decoder(x)
        
        return y