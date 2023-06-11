"""A space-time Transformer with Cuboid Attention"""
from typing import Sequence, Union
import warnings
from functools import lru_cache
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from .cuboid_transformer_patterns import CuboidSelfAttentionPatterns, CuboidCrossAttentionPatterns
from .utils import (
    get_activation, get_norm_layer,
    _generalize_padding, _generalize_unpadding,
    apply_initialization, round_to)


class PosEmbed(nn.Module):

    def __init__(self, embed_dim, maxT, maxH, maxW, typ='t+h+w'):
        r"""
        Parameters
        ----------
        embed_dim
        maxT
        maxH
        maxW
        typ
            The type of the positional embedding.
            - t+h+w:
                Embed the spatial position to embeddings
            - t+hw:
                Embed the spatial position to embeddings
        """
        super(PosEmbed, self).__init__()
        self.typ = typ

        assert self.typ in ['t+h+w', 't+hw']
        self.maxT = maxT
        self.maxH = maxH
        self.maxW = maxW
        self.embed_dim = embed_dim
        # spatiotemporal learned positional embedding
        if self.typ == 't+h+w':
            self.T_embed = nn.Embedding(num_embeddings=maxT, embedding_dim=embed_dim)
            self.H_embed = nn.Embedding(num_embeddings=maxH, embedding_dim=embed_dim)
            self.W_embed = nn.Embedding(num_embeddings=maxW, embedding_dim=embed_dim)

            # nn.init.trunc_normal_(self.T_embed.weight, std=0.02)
            # nn.init.trunc_normal_(self.H_embed.weight, std=0.02)
            # nn.init.trunc_normal_(self.W_embed.weight, std=0.02)
        elif self.typ == 't+hw':
            self.T_embed = nn.Embedding(num_embeddings=maxT, embedding_dim=embed_dim)
            self.HW_embed = nn.Embedding(num_embeddings=maxH * maxW, embedding_dim=embed_dim)
            # nn.init.trunc_normal_(self.T_embed.weight, std=0.02)
            # nn.init.trunc_normal_(self.HW_embed.weight, std=0.02)
        else:
            raise NotImplementedError
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.children():
            apply_initialization(m, embed_mode="0")

    def forward(self, x):
        """

        Parameters
        ----------
        x
            Shape (B, T, H, W, C)

        Returns
        -------
        out
            Return the x + positional embeddings
        """
        _, T, H, W, _ = x.shape
        t_idx = torch.arange(T, device=x.device)  # (T, C)
        h_idx = torch.arange(H, device=x.device)  # (H, C)
        w_idx = torch.arange(W, device=x.device)  # (W, C)
        if self.typ == 't+h+w':
            return x + self.T_embed(t_idx).reshape(T, 1, 1, self.embed_dim)\
                     + self.H_embed(h_idx).reshape(1, H, 1, self.embed_dim)\
                     + self.W_embed(w_idx).reshape(1, 1, W, self.embed_dim)
        elif self.typ == 't+hw':
            spatial_idx = h_idx.unsqueeze(-1) * self.maxW + w_idx
            return x + self.T_embed(t_idx).reshape(T, 1, 1, self.embed_dim) + self.HW_embed(spatial_idx)
        else:
            raise NotImplementedError

class PositionwiseFFN(nn.Module):
    """The Position-wise FFN layer used in Transformer-like architectures

    If pre_norm is True:
        norm(data) -> fc1 -> act -> act_dropout -> fc2 -> dropout -> res(+data)
    Else:
        data -> fc1 -> act -> act_dropout -> fc2 -> dropout -> norm(res(+data))
    Also, if we use gated projection. We will use
        fc1_1 * act(fc1_2(data)) to map the data
    """
    def __init__(self,
                 units: int = 512,
                 hidden_size: int = 2048,
                 activation_dropout: float = 0.0,
                 dropout: float = 0.1,
                 gated_proj: bool = False,
                 activation='relu',
                 normalization: str = 'layer_norm',
                 layer_norm_eps: float = 1E-5,
                 pre_norm: bool = False,
                 linear_init_mode="0",
                 norm_init_mode="0",
                 ):
        """
        Parameters
        ----------
        units
        hidden_size
        activation_dropout
        dropout
        activation
        normalization
            layer_norm or no_norm
        layer_norm_eps
        pre_norm
            Pre-layer normalization as proposed in the paper:
            "[ACL2018] The Best of Both Worlds: Combining Recent Advances in
             Neural Machine Translation"
            This will stabilize the training of Transformers.
            You may also refer to
            "[Arxiv2020] Understanding the Difficulty of Training Transformers"
        """
        super().__init__()
        # initialization
        self.linear_init_mode = linear_init_mode
        self.norm_init_mode = norm_init_mode

        self._pre_norm = pre_norm
        self._gated_proj = gated_proj
        self._kwargs = OrderedDict([
            ('units', units),
            ('hidden_size', hidden_size),
            ('activation_dropout', activation_dropout),
            ('activation', activation),
            ('dropout', dropout),
            ('normalization', normalization),
            ('layer_norm_eps', layer_norm_eps),
            ('gated_proj', gated_proj),
            ('pre_norm', pre_norm)
        ])
        self.dropout_layer = nn.Dropout(dropout)
        self.activation_dropout_layer = nn.Dropout(activation_dropout)
        self.ffn_1 = nn.Linear(in_features=units, out_features=hidden_size,
                               bias=True)
        if self._gated_proj:
            self.ffn_1_gate = nn.Linear(in_features=units,
                                        out_features=hidden_size,
                                        bias=True)
        self.activation = get_activation(activation)
        self.ffn_2 = nn.Linear(in_features=hidden_size, out_features=units,
                               bias=True)
        self.layer_norm = get_norm_layer(normalization=normalization,
                                         in_channels=units,
                                         epsilon=layer_norm_eps)
        self.reset_parameters()

    def reset_parameters(self):
        apply_initialization(self.ffn_1,
                             linear_mode=self.linear_init_mode)
        if self._gated_proj:
            apply_initialization(self.ffn_1_gate,
                                 linear_mode=self.linear_init_mode)
        apply_initialization(self.ffn_2,
                             linear_mode=self.linear_init_mode)
        apply_initialization(self.layer_norm,
                             norm_mode=self.norm_init_mode)

    def forward(self, data):
        """

        Parameters
        ----------
        data :
            Shape (B, seq_length, C_in)

        Returns
        -------
        out :
            Shape (B, seq_length, C_out)
        """
        residual = data
        if self._pre_norm:
            data = self.layer_norm(data)
        if self._gated_proj:
            out = self.activation(self.ffn_1_gate(data)) * self.ffn_1(data)
        else:
            out = self.activation(self.ffn_1(data))
        out = self.activation_dropout_layer(out)
        out = self.ffn_2(out)
        out = self.dropout_layer(out)
        out = out + residual
        if not self._pre_norm:
            out = self.layer_norm(out)
        return out

class PatchMerging3D(nn.Module):
    """ Patch Merging Layer"""
    def __init__(self,
                 dim,
                 out_dim=None,
                 downsample=(1, 2, 2),
                 norm_layer='layer_norm',
                 padding_type='nearest',
                 linear_init_mode="0",
                 norm_init_mode="0",
                 ):
        """

        Parameters
        ----------
        dim
            Number of input channels.
        downsample
            downsample factor
        norm_layer
            The normalization layer
        """
        super().__init__()
        self.linear_init_mode = linear_init_mode
        self.norm_init_mode = norm_init_mode
        self.dim = dim
        if out_dim is None:
            out_dim = max(downsample) * dim
        self.out_dim = out_dim
        self.downsample = downsample
        self.padding_type = padding_type
        self.reduction = nn.Linear(downsample[0] * downsample[1] * downsample[2] * dim,
                                   out_dim, bias=False)
        self.norm = get_norm_layer(norm_layer, in_channels=downsample[0] * downsample[1] * downsample[2] * dim)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.children():
            apply_initialization(m,
                                 linear_mode=self.linear_init_mode,
                                 norm_mode=self.norm_init_mode)

    def get_out_shape(self, data_shape):
        T, H, W, C_in = data_shape
        pad_t = (self.downsample[0] - T % self.downsample[0]) % self.downsample[0]
        pad_h = (self.downsample[1] - H % self.downsample[1]) % self.downsample[1]
        pad_w = (self.downsample[2] - W % self.downsample[2]) % self.downsample[2]
        return (T + pad_t) // self.downsample[0], (H + pad_h) // self.downsample[1], (W + pad_w) // self.downsample[2],\
               self.out_dim

    def forward(self, x):
        """

        Parameters
        ----------
        x
            Input feature, tensor size (B, T, H, W, C).

        Returns
        -------
        out
            Shape (B, T // downsample[0], H // downsample[1], W // downsample[2], out_dim)
        """
        B, T, H, W, C = x.shape

        # padding
        pad_t = (self.downsample[0] - T % self.downsample[0]) % self.downsample[0]
        pad_h = (self.downsample[1] - H % self.downsample[1]) % self.downsample[1]
        pad_w = (self.downsample[2] - W % self.downsample[2]) % self.downsample[2]
        if pad_h or pad_h or pad_w:
            T += pad_t
            H += pad_h
            W += pad_w
            x = _generalize_padding(x, pad_t, pad_h, pad_w, padding_type=self.padding_type)

        x = x.reshape((B,
                       T // self.downsample[0], self.downsample[0],
                       H // self.downsample[1], self.downsample[1],
                       W // self.downsample[2], self.downsample[2], C)) \
             .permute(0, 1, 3, 5, 2, 4, 6, 7) \
             .reshape(B, T // self.downsample[0], H // self.downsample[1], W // self.downsample[2],
                      self.downsample[0] * self.downsample[1] * self.downsample[2] * C)
        x = self.norm(x)
        x = self.reduction(x)

        return x

class Upsample3DLayer(nn.Module):
    """Upsampling based on nn.UpSampling and Conv3x3.

    If the temporal dimension remains the same:
        x --> interpolation-2d (nearest) --> conv3x3(dim, out_dim)
    Else:
        x --> interpolation-3d (nearest) --> conv3x3x3(dim, out_dim)

    """
    def __init__(self,
                 dim,
                 out_dim,
                 target_size,
                 temporal_upsample=False,
                 kernel_size=3,
                 layout='THWC',
                 conv_init_mode="0",
                 ):
        """

        Parameters
        ----------
        dim
        out_dim
        target_size
            Size of the output tensor. Will be a tuple/list that contains T_new, H_new, W_new
        temporal_upsample
            Whether the temporal axis will go through upsampling.
        kernel_size
            The kernel size of the Conv2D layer
        layout
            The layout of the inputs
        """
        super(Upsample3DLayer, self).__init__()
        self.conv_init_mode = conv_init_mode
        self.target_size = target_size
        self.out_dim = out_dim
        self.temporal_upsample = temporal_upsample
        if temporal_upsample:
            self.up = nn.Upsample(size=target_size, mode='nearest')  # 3D upsampling
        else:
            self.up = nn.Upsample(size=(target_size[1], target_size[2]), mode='nearest')  # 2D upsampling
        self.conv = nn.Conv2d(in_channels=dim, out_channels=out_dim, kernel_size=(kernel_size, kernel_size),
                              padding=(kernel_size // 2, kernel_size // 2))
        assert layout in ['THWC', 'CTHW']
        self.layout = layout

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.children():
            apply_initialization(m,
                                 conv_mode=self.conv_init_mode)

    def forward(self, x):
        """

        Parameters
        ----------
        x
            Shape (B, T, H, W, C) or (B, C, T, H, W)

        Returns
        -------
        out
            Shape (B, T, H_new, W_out, C_out) or (B, C, T, H_out, W_out)
        """
        if self.layout == 'THWC':
            B, T, H, W, C = x.shape
            if self.temporal_upsample:
                x = x.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
                return self.conv(self.up(x)).permute(0, 2, 3, 4, 1)
            else:
                assert self.target_size[0] == T
                x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)  # (B * T, C, H, W)
                x = self.up(x)
                return self.conv(x).permute(0, 2, 3, 1).reshape((B,) + self.target_size + (self.out_dim,))
        elif self.layout == 'CTHW':
            B, C, T, H, W = x.shape
            if self.temporal_upsample:
                return self.conv(self.up(x))
            else:
                assert self.output_size[0] == T
                x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
                x = x.reshape(B * T, C, H, W)
                return self.conv(self.up(x)).reshape(B, self.target_size[0], self.out_dim, self.target_size[1],
                                                     self.target_size[2]).permute(0, 2, 1, 3, 4)

def cuboid_reorder(data, cuboid_size, strategy):
    """Reorder the tensor into (B, num_cuboids, bT * bH * bW, C)

    We assume that the tensor shapes are divisible to the cuboid sizes.

    Parameters
    ----------
    data
        The input data
    cuboid_size
        The size of the cuboid
    strategy
        The cuboid strategy

    Returns
    -------
    reordered_data
        Shape will be (B, num_cuboids, bT * bH * bW, C)
        num_cuboids = T / bT * H / bH * W / bW
    """
    B, T, H, W, C = data.shape
    num_cuboids = T // cuboid_size[0] * H // cuboid_size[1] * W // cuboid_size[2]
    cuboid_volume = cuboid_size[0] * cuboid_size[1] * cuboid_size[2]
    intermediate_shape = []

    nblock_axis = []
    block_axis = []
    for i, (block_size, total_size, ele_strategy) in enumerate(zip(cuboid_size, (T, H, W), strategy)):
        if ele_strategy == 'l':
            intermediate_shape.extend([total_size // block_size, block_size])
            nblock_axis.append(2 * i + 1)
            block_axis.append(2 * i + 2)
        elif ele_strategy == 'd':
            intermediate_shape.extend([block_size, total_size // block_size])
            nblock_axis.append(2 * i + 2)
            block_axis.append(2 * i + 1)
        else:
            raise NotImplementedError
    data = data.reshape((B,) + tuple(intermediate_shape) + (C, ))
    reordered_data = data.permute((0,) + tuple(nblock_axis) + tuple(block_axis) + (7,))
    reordered_data = reordered_data.reshape((B, num_cuboids, cuboid_volume, C))
    return reordered_data


def cuboid_reorder_reverse(data, cuboid_size, strategy, orig_data_shape):
    """Reverse the reordered cuboid back to the original space

    Parameters
    ----------
    data
    cuboid_size
    strategy
    orig_data_shape

    Returns
    -------
    data
        The recovered data
    """
    B, num_cuboids, cuboid_volume, C = data.shape
    T, H, W = orig_data_shape

    permutation_axis = [0]
    for i, (block_size, total_size, ele_strategy) in enumerate(zip(cuboid_size, (T, H, W), strategy)):
        if ele_strategy == 'l':
            # intermediate_shape.extend([total_size // block_size, block_size])
            permutation_axis.append(i + 1)
            permutation_axis.append(i + 4)
        elif ele_strategy == 'd':
            # intermediate_shape.extend([block_size, total_size // block_size])
            permutation_axis.append(i + 4)
            permutation_axis.append(i + 1)
        else:
            raise NotImplementedError
    permutation_axis.append(7)
    data = data.reshape(B, T // cuboid_size[0], H // cuboid_size[1], W // cuboid_size[2],
                        cuboid_size[0], cuboid_size[1], cuboid_size[2], C)
    data = data.permute(permutation_axis)
    data = data.reshape((B, T, H, W, C))
    return data


@lru_cache()
def compute_cuboid_self_attention_mask(data_shape, cuboid_size, shift_size, strategy, padding_type, device):
    """Compute the shift window attention mask

    Parameters
    ----------
    data_shape
        Should be T, H, W
    cuboid_size
        Size of the cuboid
    shift_size
        The shift size
    strategy
        The decomposition strategy
    padding_type
        Type of the padding
    device
        The device

    Returns
    -------
    attn_mask
        Mask with shape (num_cuboid, cuboid_vol, cuboid_vol)
        The padded values will always be masked. The other masks will ensure that the shifted windows
        will only attend to those in the shifted windows.
    """
    T, H, W = data_shape
    pad_t = (cuboid_size[0] - T % cuboid_size[0]) % cuboid_size[0]
    pad_h = (cuboid_size[1] - H % cuboid_size[1]) % cuboid_size[1]
    pad_w = (cuboid_size[2] - W % cuboid_size[2]) % cuboid_size[2]
    data_mask = None
    # Prepare data mask
    if pad_t > 0  or pad_h > 0 or pad_w > 0:
        if padding_type == 'ignore':
            data_mask = torch.ones((1, T, H, W, 1), dtype=torch.bool, device=device)
            data_mask = F.pad(data_mask, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t))
    else:
        data_mask = torch.ones((1, T + pad_t, H + pad_h, W + pad_w, 1), dtype=torch.bool, device=device)
    if any(i > 0 for i in shift_size):
        if padding_type == 'ignore':
            data_mask = torch.roll(data_mask,
                                   shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
    if padding_type == 'ignore':
        # (1, num_cuboids, cuboid_volume, 1)
        data_mask = cuboid_reorder(data_mask, cuboid_size, strategy=strategy)
        data_mask = data_mask.squeeze(-1).squeeze(0)  # (num_cuboid, cuboid_volume)
    # Prepare mask based on index
    shift_mask = torch.zeros((1, T + pad_t, H + pad_h, W + pad_w, 1), device=device)  # 1 T H W 1
    cnt = 0
    for t in slice(-cuboid_size[0]), slice(-cuboid_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-cuboid_size[1]), slice(-cuboid_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-cuboid_size[2]), slice(-cuboid_size[2], -shift_size[2]), slice(-shift_size[2], None):
                shift_mask[:, t, h, w, :] = cnt
                cnt += 1
    shift_mask = cuboid_reorder(shift_mask, cuboid_size, strategy=strategy)
    shift_mask = shift_mask.squeeze(-1).squeeze(0)  # num_cuboids, cuboid_volume
    attn_mask = (shift_mask.unsqueeze(1) - shift_mask.unsqueeze(2)) == 0  # num_cuboids, cuboid_volume, cuboid_volume
    if padding_type == 'ignore':
        attn_mask = data_mask.unsqueeze(1) * data_mask.unsqueeze(2) * attn_mask
    return attn_mask

def masked_softmax(att_score, mask, axis: int = -1):
    """Ignore the masked elements when calculating the softmax.
     The mask can be broadcastable.

    Parameters
    ----------
    att_score
        Shape (..., length, ...)
    mask
        Shape (..., length, ...)
        1 --> The element is not masked
        0 --> The element is masked
    axis
        The axis to calculate the softmax. att_score.shape[axis] must be the same as mask.shape[axis]

    Returns
    -------
    att_weights
        Shape (..., length, ...)
    """
    if mask is not None:
        # Fill in the masked scores with a very small value
        if att_score.dtype == torch.float16:
            att_score = att_score.masked_fill(torch.logical_not(mask), -1E4)
        else:
            att_score = att_score.masked_fill(torch.logical_not(mask), -1E18)
        att_weights = torch.softmax(att_score, dim=axis) * mask
    else:
        att_weights = torch.softmax(att_score, dim=axis)
    return att_weights

def update_cuboid_size_shift_size(data_shape, cuboid_size, shift_size, strategy):
    """Update the

    Parameters
    ----------
    data_shape
        The shape of the data
    cuboid_size
        Size of the cuboid
    shift_size
        Size of the shift
    strategy
        The strategy of attention

    Returns
    -------
    new_cuboid_size
        Size of the cuboid
    new_shift_size
        Size of the shift
    """
    new_cuboid_size = list(cuboid_size)
    new_shift_size = list(shift_size)
    for i in range(len(data_shape)):
        if strategy[i] == 'd':
            new_shift_size[i] = 0
        if data_shape[i] <= cuboid_size[i]:
            new_cuboid_size[i] = data_shape[i]
            new_shift_size[i] = 0
    return tuple(new_cuboid_size), tuple(new_shift_size)

class CuboidSelfAttentionLayer(nn.Module):
    """Implements the cuboid self attention.

    The idea of Cuboid Self Attention is to divide the input tensor (T, H, W) into several non-overlapping cuboids.
    We apply self-attention inside each cuboid and all cuboid-level self attentions are executed in parallel.

    We adopt two mechanisms for decomposing the input tensor into cuboids:

    1) local:
        We group the tensors within a local window, e.g., X[t:(t+b_t), h:(h+b_h), w:(w+b_w)]. We can also apply the
        shifted window strategy proposed in "[ICCV2021] Swin Transformer: Hierarchical Vision Transformer using Shifted Windows".
    2) dilated:
        Inspired by the success of dilated convolution "[ICLR2016] Multi-Scale Context Aggregation by Dilated Convolutions",
         we split the tensor with dilation factors that are tied to the size of the cuboid. For example, for a cuboid that has width `b_w`,
         we sample the elements starting from 0 as 0, w / b_w, 2 * w / b_w, ..., (b_w - 1) * w / b_w.

    The cuboid attention can be viewed as a generalization of the attention mechanism proposed in Video Swin Transformer, https://arxiv.org/abs/2106.13230.
    The computational complexity of CuboidAttention can be simply calculated as O(T H W * b_t b_h b_w). To cover multiple correlation patterns,
    we are able to combine multiple CuboidAttention layers with different configurations such as cuboid size, shift size, and local / global decomposing strategy.

    In addition, it is straight-forward to extend the cuboid attention to other types of spatiotemporal data that are not described
    as regular tensors. We need to define alternative approaches to partition the data into "cuboids".

    In addition, inspired by "[NeurIPS2021] Do Transformers Really Perform Badly for Graph Representation?",
     "[NeurIPS2020] Big Bird: Transformers for Longer Sequences", "[EMNLP2021] Longformer: The Long-Document Transformer", we keep
     $K$ global vectors to record the global status of the spatiotemporal system. These global vectors will attend to the whole tensor and
     the vectors inside each individual cuboids will also attend to the global vectors so that they can peep into the global status of the system.

    """
    def __init__(self,
                 dim,
                 num_heads,
                 cuboid_size=(2, 7, 7),
                 shift_size=(0, 0, 0),
                 strategy=('l', 'l', 'l'),
                 padding_type='ignore',
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 use_final_proj=True,
                 norm_layer='layer_norm',
                 use_global_vector=False,
                 use_global_self_attn=False,
                 separate_global_qkv=False,
                 global_dim_ratio=1,
                 checkpoint_level=True,
                 use_relative_pos=True,
                 attn_linear_init_mode="0",
                 ffn_linear_init_mode="0",
                 norm_init_mode="0",
                 ):
        """

        Parameters
        ----------
        dim
            The dimension of the input tensor
        num_heads
            The number of heads
        cuboid_size
            The size of each cuboid
        shift_size
            The size for shifting the windows.
        strategy
            The decomposition strategy of the tensor. 'l' stands for local and 'd' stands for dilated.
        padding_type
            The type of padding.
        qkv_bias
            Whether to enable bias in calculating qkv attention
        qk_scale
            Whether to enable scale factor when calculating the attention.
        attn_drop
            The attention dropout
        proj_drop
            The projection dropout
        use_final_proj
            Whether to use the final projection or not
        norm_layer
            The normalization layer
        use_global_vector
            Whether to use the global vector or not.
        use_global_self_attn
            Whether to do self attention among global vectors
        separate_global_qkv
            Whether to different network to calc q_global, k_global, v_global
        global_dim_ratio
            The dim (channels) of global vectors is `global_dim_ratio*dim`.
        checkpoint_level
            Whether to enable gradient checkpointing.
        """
        super(CuboidSelfAttentionLayer, self).__init__()
        # initialization
        self.attn_linear_init_mode = attn_linear_init_mode
        self.ffn_linear_init_mode = ffn_linear_init_mode
        self.norm_init_mode = norm_init_mode

        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.dim = dim
        self.cuboid_size = cuboid_size
        self.shift_size = shift_size
        self.strategy = strategy
        self.padding_type = padding_type
        self.use_final_proj = use_final_proj
        self.use_relative_pos = use_relative_pos
        # global vectors
        self.use_global_vector = use_global_vector
        self.use_global_self_attn = use_global_self_attn
        self.separate_global_qkv = separate_global_qkv
        if global_dim_ratio != 1:
            assert separate_global_qkv == True, \
                f"Setting global_dim_ratio != 1 requires separate_global_qkv == True."
        self.global_dim_ratio = global_dim_ratio

        assert self.padding_type in ['ignore', 'zeros', 'nearest']
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if use_relative_pos:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * cuboid_size[0] - 1) * (2 * cuboid_size[1] - 1) * (2 * cuboid_size[2] - 1), num_heads))
            nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

            coords_t = torch.arange(self.cuboid_size[0])
            coords_h = torch.arange(self.cuboid_size[1])
            coords_w = torch.arange(self.cuboid_size[2])
            coords = torch.stack(torch.meshgrid(coords_t, coords_h, coords_w))  # 3, Bt, Bh, Bw

            coords_flatten = torch.flatten(coords, 1)  # 3, Bt*Bh*Bw
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Bt*Bh*Bw, Bt*Bh*Bw
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Bt*Bh*Bw, Bt*Bh*Bw, 3
            relative_coords[:, :, 0] += self.cuboid_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.cuboid_size[1] - 1
            relative_coords[:, :, 2] += self.cuboid_size[2] - 1

            relative_coords[:, :, 0] *= (2 * self.cuboid_size[1] - 1) * (2 * self.cuboid_size[2] - 1)
            relative_coords[:, :, 1] *= (2 * self.cuboid_size[2] - 1)
            relative_position_index = relative_coords.sum(-1)  # shape is (cuboid_volume, cuboid_volume)
            self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        if self.use_global_vector:
            if self.separate_global_qkv:
                self.l2g_q_net = nn.Linear(dim, dim, bias=qkv_bias)
                self.l2g_global_kv_net = nn.Linear(
                    in_features=global_dim_ratio * dim,
                    out_features=dim * 2,
                    bias=qkv_bias)
                self.g2l_global_q_net = nn.Linear(
                    in_features=global_dim_ratio * dim,
                    out_features=dim,
                    bias=qkv_bias)
                self.g2l_k_net = nn.Linear(
                    in_features=dim,
                    out_features=dim,
                    bias=qkv_bias)
                self.g2l_v_net = nn.Linear(
                    in_features=dim,
                    out_features=global_dim_ratio * dim,
                    bias=qkv_bias)
                if self.use_global_self_attn:
                    self.g2g_global_qkv_net = nn.Linear(
                        in_features=global_dim_ratio * dim,
                        out_features=global_dim_ratio * dim * 3,
                        bias=qkv_bias)
            else:
                self.global_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.global_attn_drop = nn.Dropout(attn_drop)

        if use_final_proj:
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)

            if self.use_global_vector:
                self.global_proj = nn.Linear(
                    in_features=global_dim_ratio * dim,
                    out_features=global_dim_ratio * dim)

        self.norm = get_norm_layer(norm_layer, in_channels=dim)
        if self.use_global_vector:
            self.global_vec_norm = get_norm_layer(norm_layer,
                                                  in_channels=global_dim_ratio*dim)

        self.checkpoint_level = checkpoint_level
        self.reset_parameters()

    def reset_parameters(self):
        apply_initialization(self.qkv,
                             linear_mode=self.attn_linear_init_mode)
        if self.use_final_proj:
            apply_initialization(self.proj,
                                 linear_mode=self.ffn_linear_init_mode)
        apply_initialization(self.norm,
                             norm_mode=self.norm_init_mode)
        if self.use_global_vector:
            if self.separate_global_qkv:
                apply_initialization(self.l2g_q_net,
                                     linear_mode=self.attn_linear_init_mode)
                apply_initialization(self.l2g_global_kv_net,
                                     linear_mode=self.attn_linear_init_mode)
                apply_initialization(self.g2l_global_q_net,
                                     linear_mode=self.attn_linear_init_mode)
                apply_initialization(self.g2l_k_net,
                                     linear_mode=self.attn_linear_init_mode)
                apply_initialization(self.g2l_v_net,
                                     linear_mode=self.attn_linear_init_mode)
                if self.use_global_self_attn:
                    apply_initialization(self.g2g_global_qkv_net,
                                         linear_mode=self.attn_linear_init_mode)
            else:
                apply_initialization(self.global_qkv,
                                     linear_mode=self.attn_linear_init_mode)
            apply_initialization(self.global_vec_norm,
                                 norm_mode=self.norm_init_mode)

    def forward(self, x, global_vectors=None):
        x = self.norm(x)

        B, T, H, W, C_in = x.shape
        assert C_in == self.dim
        if self.use_global_vector:
            _, num_global, _ = global_vectors.shape
            global_vectors = self.global_vec_norm(global_vectors)

        cuboid_size, shift_size = update_cuboid_size_shift_size((T, H, W), self.cuboid_size,
                                                                self.shift_size, self.strategy)
        # Step-1: Pad the input
        pad_t = (cuboid_size[0] - T % cuboid_size[0]) % cuboid_size[0]
        pad_h = (cuboid_size[1] - H % cuboid_size[1]) % cuboid_size[1]
        pad_w = (cuboid_size[2] - W % cuboid_size[2]) % cuboid_size[2]

        # We use generalized padding
        x = _generalize_padding(x, pad_t, pad_h, pad_w, self.padding_type)

        # Step-2: Shift the tensor based on shift window attention.

        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
        else:
            shifted_x = x
        # Step-3: Reorder the tensor
        # (B, num_cuboids, cuboid_volume, C)
        reordered_x = cuboid_reorder(shifted_x, cuboid_size=cuboid_size, strategy=self.strategy)
        _, num_cuboids, cuboid_volume, _ = reordered_x.shape
        # Step-4: Perform self-attention
        # (num_cuboids, cuboid_volume, cuboid_volume)
        attn_mask = compute_cuboid_self_attention_mask((T, H, W), cuboid_size,
                                                       shift_size=shift_size,
                                                       strategy=self.strategy,
                                                       padding_type=self.padding_type,
                                                       device=x.device)
        head_C = C_in // self.num_heads
        qkv = self.qkv(reordered_x).reshape(B, num_cuboids, cuboid_volume, 3, self.num_heads, head_C)\
            .permute(3, 0, 4, 1, 2, 5)  # (3, B, num_heads, num_cuboids, cuboid_volume, head_C)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each has shape (B, num_heads, num_cuboids, cuboid_volume, head_C)
        q = q * self.scale
        attn_score = q @ k.transpose(-2, -1)  # Shape (B, num_heads, num_cuboids, cuboid_volume, cuboid_volume)

        if self.use_relative_pos:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index[:cuboid_volume, :cuboid_volume].reshape(-1)]\
                .reshape(cuboid_volume, cuboid_volume, -1)  # (cuboid_volume, cuboid_volume, num_head)
            relative_position_bias = relative_position_bias.permute(2, 0, 1)\
                .contiguous().unsqueeze(1)  # num_heads, 1, cuboid_volume, cuboid_volume
            attn_score = attn_score + relative_position_bias  # Shape (B, num_heads, num_cuboids, cuboid_volume, cuboid_volume)

        # Calculate the local to global attention
        if self.use_global_vector:
            global_head_C = self.global_dim_ratio * head_C # take effect only separate_global_qkv = True
            if self.separate_global_qkv:
                l2g_q = self.l2g_q_net(reordered_x)\
                    .reshape(B, num_cuboids, cuboid_volume, self.num_heads, head_C)\
                    .permute(0, 3, 1, 2, 4)  # (B, num_heads, num_cuboids, cuboid_volume, head_C)
                l2g_q = l2g_q * self.scale
                l2g_global_kv = self.l2g_global_kv_net(global_vectors)\
                    .reshape(B, 1, num_global, 2, self.num_heads, head_C)\
                    .permute(3, 0, 4, 1, 2, 5)  # Shape (2, B, num_heads, 1, N, head_C)
                l2g_global_k, l2g_global_v = l2g_global_kv[0], l2g_global_kv[1]
                g2l_global_q = self.g2l_global_q_net(global_vectors)\
                    .reshape(B, num_global, self.num_heads, head_C)\
                    .permute(0, 2, 1, 3)  # Shape (B, num_heads, N, head_C)
                g2l_global_q = g2l_global_q * self.scale
                # g2l_kv = self.g2l_kv_net(reordered_x)\
                #     .reshape(B, num_cuboids, cuboid_volume, 2, self.num_heads, global_head_C)\
                #     .permute(3, 0, 4, 1, 2, 5)  # (2, B, num_heads, num_cuboids, cuboid_volume, head_C)
                # g2l_k, g2l_v = g2l_kv[0], g2l_kv[1]
                g2l_k = self.g2l_k_net(reordered_x)\
                    .reshape(B, num_cuboids, cuboid_volume, self.num_heads, head_C)\
                    .permute(0, 3, 1, 2, 4)  # (B, num_heads, num_cuboids, cuboid_volume, head_C)
                g2l_v = self.g2l_v_net(reordered_x) \
                    .reshape(B, num_cuboids, cuboid_volume, self.num_heads, global_head_C) \
                    .permute(0, 3, 1, 2, 4)  # (B, num_heads, num_cuboids, cuboid_volume, global_head_C)
                if self.use_global_self_attn:
                    g2g_global_qkv = self.g2g_global_qkv_net(global_vectors)\
                    .reshape(B, 1, num_global, 3, self.num_heads, global_head_C)\
                    .permute(3, 0, 4, 1, 2, 5)  # Shape (2, B, num_heads, 1, N, head_C)
                    g2g_global_q, g2g_global_k, g2g_global_v = g2g_global_qkv[0], g2g_global_qkv[1], g2g_global_qkv[2]
                    g2g_global_q = g2g_global_q.squeeze(2) * self.scale
            else:
                q_global, k_global, v_global = self.global_qkv(global_vectors)\
                    .reshape(B, 1, num_global, 3, self.num_heads, head_C)\
                    .permute(3, 0, 4, 1, 2, 5)  # Shape (3, B, num_heads, 1, N, head_C)
                q_global = q_global.squeeze(2) * self.scale
                l2g_q, g2l_k, g2l_v = q, k, v
                g2l_global_q, l2g_global_k, l2g_global_v = q_global, k_global, v_global
                if self.use_global_self_attn:
                    g2g_global_q, g2g_global_k, g2g_global_v = q_global, k_global, v_global
            l2g_attn_score = l2g_q @ l2g_global_k.transpose(-2, -1)  # Shape (B, num_heads, num_cuboids, cuboid_volume, N)
            attn_score_l2l_l2g = torch.cat((attn_score, l2g_attn_score),
                                           dim=-1)  # Shape (B, num_heads, num_cuboids, cuboid_volume, cuboid_volume + N)
            attn_mask_l2l_l2g = F.pad(attn_mask, (0, num_global), "constant", 1)
            v_l_g = torch.cat((v, l2g_global_v.expand(B, self.num_heads, num_cuboids, num_global, head_C)),
                            dim=3)
            # local to local and global attention
            attn_score_l2l_l2g = masked_softmax(attn_score_l2l_l2g, mask=attn_mask_l2l_l2g)
            attn_score_l2l_l2g = self.attn_drop(attn_score_l2l_l2g)  # Shape (B, num_heads, num_cuboids, x_cuboid_volume, mem_cuboid_volume + K))
            reordered_x = (attn_score_l2l_l2g @ v_l_g).permute(0, 2, 3, 1, 4) \
                .reshape(B, num_cuboids, cuboid_volume, self.dim)
            # update global vectors
            if self.padding_type == 'ignore':
                g2l_attn_mask = torch.ones((1, T, H, W, 1), device=x.device)
                if pad_t > 0 or pad_h > 0 or pad_w > 0:
                    g2l_attn_mask = F.pad(g2l_attn_mask, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t))
                if any(i > 0 for i in shift_size):
                    g2l_attn_mask = torch.roll(g2l_attn_mask, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
                                               dims=(1, 2, 3))
                g2l_attn_mask = g2l_attn_mask.reshape((-1,))
            else:
                g2l_attn_mask = None
            g2l_attn_score = g2l_global_q @ g2l_k.reshape(B, self.num_heads, num_cuboids * cuboid_volume, head_C).transpose(-2, -1)  # Shape (B, num_heads, N, num_cuboids * cuboid_volume)
            if self.use_global_self_attn:
                g2g_attn_score = g2g_global_q @ g2g_global_k.squeeze(2).transpose(-2, -1)
                g2all_attn_score = torch.cat((g2l_attn_score, g2g_attn_score),
                                             dim=-1)  # Shape (B, num_heads, N, num_cuboids * cuboid_volume + N)
                if g2l_attn_mask is not None:
                    g2all_attn_mask = F.pad(g2l_attn_mask, (0, num_global), "constant", 1)
                else:
                    g2all_attn_mask = None
                new_v = torch.cat((g2l_v.reshape(B, self.num_heads, num_cuboids * cuboid_volume, global_head_C),
                                   g2g_global_v.reshape(B, self.num_heads, num_global, global_head_C)),
                                  dim=2)
            else:
                g2all_attn_score = g2l_attn_score
                g2all_attn_mask = g2l_attn_mask
                new_v = g2l_v.reshape(B, self.num_heads, num_cuboids * cuboid_volume, global_head_C)
            g2all_attn_score = masked_softmax(g2all_attn_score, mask=g2all_attn_mask)
            g2all_attn_score = self.global_attn_drop(g2all_attn_score)
            new_global_vector = (g2all_attn_score @ new_v).permute(0, 2, 1, 3).\
                reshape(B, num_global, self.global_dim_ratio*self.dim)
        else:
            attn_score = masked_softmax(attn_score, mask=attn_mask)
            attn_score = self.attn_drop(attn_score)  # Shape (B, num_heads, num_cuboids, cuboid_volume, cuboid_volume (+ K))
            reordered_x = (attn_score @ v).permute(0, 2, 3, 1, 4).reshape(B, num_cuboids, cuboid_volume, self.dim)

        if self.use_final_proj:
            reordered_x = self.proj_drop(self.proj(reordered_x))
            if self.use_global_vector:
                new_global_vector = self.proj_drop(self.global_proj(new_global_vector))
        # Step-5: Shift back and slice
        shifted_x = cuboid_reorder_reverse(reordered_x, cuboid_size=cuboid_size, strategy=self.strategy,
                                           orig_data_shape=(T + pad_t, H + pad_h, W + pad_w))
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x
        x = _generalize_unpadding(x, pad_t=pad_t, pad_h=pad_h, pad_w=pad_w, padding_type=self.padding_type)
        if self.use_global_vector:
            return x, new_global_vector
        else:
            return x

class StackCuboidSelfAttentionBlock(nn.Module):
    """

    - "use_inter_ffn" is True
        x --> attn1 -----+-------> ffn1 ---+---> attn2 --> ... --> ffn_k --> out
           |             ^   |             ^
           |             |   |             |
           |-------------|   |-------------|
    - "use_inter_ffn" is False
        x --> attn1 -----+------> attn2 --> ... attnk --+----> ffnk ---+---> out
           |             ^   |            ^             ^  |           ^
           |             |   |            |             |  |           |
           |-------------|   |------------|   ----------|  |-----------|
    If we have enabled global memory vectors, each attention will be a

    """
    def __init__(self,
                 dim,
                 num_heads,
                 block_cuboid_size=[(4, 4, 4), (4, 4, 4)],
                 block_shift_size=[(0, 0, 0), (2, 2, 2)],
                 block_strategy=[('d', 'd', 'd'),
                                 ('l', 'l', 'l')],
                 padding_type='ignore',
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 ffn_drop=0.0,
                 activation='leaky',
                 gated_ffn=False,
                 norm_layer='layer_norm',
                 use_inter_ffn=False,
                 use_global_vector=False,
                 use_global_vector_ffn=True,
                 use_global_self_attn=False,
                 separate_global_qkv=False,
                 global_dim_ratio=1,
                 checkpoint_level=True,
                 use_relative_pos=True,
                 use_final_proj=True,
                 # initialization
                 attn_linear_init_mode="0",
                 ffn_linear_init_mode="0",
                 norm_init_mode="0",
                 ):
        super(StackCuboidSelfAttentionBlock, self).__init__()
        # initialization
        self.attn_linear_init_mode = attn_linear_init_mode
        self.ffn_linear_init_mode = ffn_linear_init_mode
        self.norm_init_mode = norm_init_mode

        assert len(block_cuboid_size[0]) > 0 and len(block_shift_size) > 0 and len(block_strategy) > 0,\
            f'Format of the block cuboid size is not correct.' \
            f' block_cuboid_size={block_cuboid_size}'
        assert len(block_cuboid_size) == len(block_shift_size) == len(block_strategy)
        self.num_attn = len(block_cuboid_size)
        self.checkpoint_level = checkpoint_level
        self.use_inter_ffn = use_inter_ffn
        # global vectors
        self.use_global_vector = use_global_vector
        self.use_global_vector_ffn = use_global_vector_ffn
        self.use_global_self_attn = use_global_self_attn
        self.global_dim_ratio = global_dim_ratio

        if self.use_inter_ffn:
            self.ffn_l = nn.ModuleList(
                [PositionwiseFFN(
                    units=dim,
                    hidden_size=4 * dim,
                    activation_dropout=ffn_drop,
                    dropout=ffn_drop,
                    gated_proj=gated_ffn,
                    activation=activation,
                    normalization=norm_layer,
                    pre_norm=True,
                    linear_init_mode=ffn_linear_init_mode,
                    norm_init_mode=norm_init_mode,)
                    for _ in range(self.num_attn)])
            if self.use_global_vector_ffn and self.use_global_vector:
                self.global_ffn_l = nn.ModuleList(
                    [PositionwiseFFN(
                        units=global_dim_ratio * dim,
                        hidden_size=global_dim_ratio * 4 * dim,
                        activation_dropout=ffn_drop,
                        dropout=ffn_drop,
                        gated_proj=gated_ffn,
                        activation=activation,
                        normalization=norm_layer,
                        pre_norm=True,
                        linear_init_mode=ffn_linear_init_mode,
                        norm_init_mode=norm_init_mode,)
                        for _ in range(self.num_attn)])
        else:
            self.ffn_l = nn.ModuleList(
                [PositionwiseFFN(
                    units=dim, hidden_size=4 * dim,
                    activation_dropout=ffn_drop,
                    dropout=ffn_drop,
                    gated_proj=gated_ffn, activation=activation,
                    normalization=norm_layer,
                    pre_norm=True,
                    linear_init_mode=ffn_linear_init_mode,
                    norm_init_mode=norm_init_mode,)])
            if self.use_global_vector_ffn and self.use_global_vector:
                self.global_ffn_l = nn.ModuleList(
                    [PositionwiseFFN(
                        units=global_dim_ratio * dim,
                        hidden_size=global_dim_ratio * 4 * dim,
                        activation_dropout=ffn_drop,
                        dropout=ffn_drop,
                        gated_proj=gated_ffn, activation=activation,
                        normalization=norm_layer,
                        pre_norm=True,
                        linear_init_mode=ffn_linear_init_mode,
                        norm_init_mode=norm_init_mode,)])
        self.attn_l = nn.ModuleList(
            [CuboidSelfAttentionLayer(
                dim=dim, num_heads=num_heads,
                cuboid_size=ele_cuboid_size,
                shift_size=ele_shift_size,
                strategy=ele_strategy,
                padding_type=padding_type,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                norm_layer=norm_layer,
                use_global_vector=use_global_vector,
                use_global_self_attn=use_global_self_attn,
                separate_global_qkv=separate_global_qkv,
                global_dim_ratio=global_dim_ratio,
                checkpoint_level=checkpoint_level,
                use_relative_pos=use_relative_pos,
                use_final_proj=use_final_proj,
                attn_linear_init_mode=attn_linear_init_mode,
                ffn_linear_init_mode=ffn_linear_init_mode,
                norm_init_mode=norm_init_mode,)
                for ele_cuboid_size, ele_shift_size, ele_strategy
                in zip(block_cuboid_size, block_shift_size, block_strategy)])

    def reset_parameters(self):
        for m in self.ffn_l:
            m.reset_parameters()
        if self.use_global_vector_ffn and self.use_global_vector:
            for m in self.global_ffn_l:
                m.reset_parameters()
        for m in self.attn_l:
            m.reset_parameters()

    def forward(self, x, global_vectors=None):
        if self.use_inter_ffn:
            if self.use_global_vector:
                for idx, (attn, ffn) in enumerate(zip(self.attn_l, self.ffn_l)):
                    if self.checkpoint_level >= 2 and self.training:
                        x_out, global_vectors_out = checkpoint.checkpoint(attn, x, global_vectors)
                    else:
                        x_out, global_vectors_out = attn(x, global_vectors)
                    x = x + x_out
                    global_vectors = global_vectors + global_vectors_out

                    if self.checkpoint_level >= 1 and self.training:
                        x = checkpoint.checkpoint(ffn, x)
                        if self.use_global_vector_ffn:
                            global_vectors = checkpoint.checkpoint(self.global_ffn_l[idx], global_vectors)
                    else:
                        x = ffn(x)
                        if self.use_global_vector_ffn:
                            global_vectors = self.global_ffn_l[idx](global_vectors)
                return x, global_vectors
            else:
                for idx, (attn, ffn) in enumerate(zip(self.attn_l, self.ffn_l)):
                    if self.checkpoint_level >= 2 and self.training:
                        x = x + checkpoint.checkpoint(attn, x)
                    else:
                        x = x + attn(x)
                    if self.checkpoint_level >= 1 and self.training:
                        x = checkpoint.checkpoint(ffn, x)
                    else:
                        x = ffn(x)
                return x
        else:
            if self.use_global_vector:
                for idx, attn in enumerate(self.attn_l):
                    if self.checkpoint_level >= 2 and self.training:
                        x_out, global_vectors_out = checkpoint.checkpoint(attn, x, global_vectors)
                    else:
                        x_out, global_vectors_out = attn(x, global_vectors)
                    x = x + x_out
                    global_vectors = global_vectors + global_vectors_out
                if self.checkpoint_level >= 1 and self.training:
                    x = checkpoint.checkpoint(self.ffn_l[0], x)
                    if self.use_global_vector_ffn:
                        global_vectors = checkpoint.checkpoint(self.global_ffn_l[0], global_vectors)
                else:
                    x = self.ffn_l[0](x)
                    if self.use_global_vector_ffn:
                        global_vectors = self.global_ffn_l[0](global_vectors)
                return x, global_vectors
            else:
                for idx, attn in enumerate(self.attn_l):
                    if self.checkpoint_level >= 2 and self.training:
                        out = checkpoint.checkpoint(attn, x)
                    else:
                        out = attn(x)
                    x = x + out
                if self.checkpoint_level >= 1 and self.training:
                    x = checkpoint.checkpoint(self.ffn_l[0], x)
                else:
                    x = self.ffn_l[0](x)
                return x

@lru_cache()
def compute_cuboid_cross_attention_mask(T_x, T_mem, H, W, n_temporal, cuboid_hw, shift_hw, strategy,
                                        padding_type, device):
    """

    Parameters
    ----------
    T_x
    T_mem
    H
    W
    n_temporal
    cuboid_hw
    shift_hw
    strategy
    padding_type
    device

    Returns
    -------
    attn_mask
        Mask with shape (num_cuboid, x_cuboid_vol, mem_cuboid_vol)
        The padded values will always be masked. The other masks will ensure that the shifted windows
        will only attend to those in the shifted windows.
    """
    pad_t_mem = (n_temporal - T_mem % n_temporal) % n_temporal
    pad_t_x = (n_temporal - T_x % n_temporal) % n_temporal
    pad_h = (cuboid_hw[0] - H % cuboid_hw[0]) % cuboid_hw[0]
    pad_w = (cuboid_hw[1] - W % cuboid_hw[1]) % cuboid_hw[1]

    mem_cuboid_size = ((T_mem + pad_t_mem) // n_temporal,) + cuboid_hw
    x_cuboid_size = ((T_x + pad_t_x) // n_temporal,) + cuboid_hw
    if pad_t_mem > 0 or pad_h > 0 or pad_w > 0:
        if padding_type == 'ignore':
            mem_mask = torch.ones((1, T_mem, H, W, 1), dtype=torch.bool, device=device)
            mem_mask = F.pad(mem_mask, (0, 0, 0, pad_w, 0, pad_h, pad_t_mem, 0))
    else:
        mem_mask = torch.ones((1, T_mem + pad_t_mem, H + pad_h, W + pad_w, 1), dtype=torch.bool, device=device)
    if pad_t_x > 0 or pad_h > 0 or pad_w > 0:
        if padding_type == 'ignore':
            x_mask = torch.ones((1, T_x, H, W, 1), dtype=torch.bool, device=device)
            x_mask = F.pad(x_mask, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t_x))
    else:
        x_mask = torch.ones((1, T_x + pad_t_x, H + pad_h, W + pad_w, 1), dtype=torch.bool, device=device)

    if any(i > 0 for i in shift_hw):
        if padding_type == 'ignore':
            x_mask = torch.roll(x_mask, shifts=(-shift_hw[0], -shift_hw[1]), dims=(2, 3))
            mem_mask = torch.roll(mem_mask, shifts=(-shift_hw[0], -shift_hw[1]), dims=(2, 3))
    # (1, num_cuboids, cuboid_volume, 1)
    x_mask = cuboid_reorder(x_mask, x_cuboid_size, strategy=strategy)
    x_mask = x_mask.squeeze(-1).squeeze(0)  # (num_cuboid, x_cuboid_volume)
    num_cuboids, x_cuboid_volume = x_mask.shape
    mem_mask = cuboid_reorder(mem_mask, mem_cuboid_size, strategy=strategy)
    mem_mask = mem_mask.squeeze(-1).squeeze(0)  # (num_cuboid, mem_cuboid_volume)
    _, mem_cuboid_volume = mem_mask.shape

    # Prepare mask based on index
    shift_mask = torch.zeros((1, n_temporal, H + pad_h, W + pad_w, 1), device=device)  # 1 1 H W 1

    cnt = 0
    for h in slice(-cuboid_hw[0]), slice(-cuboid_hw[0], -shift_hw[0]), slice(-shift_hw[0], None):
        for w in slice(-cuboid_hw[1]), slice(-cuboid_hw[1], -shift_hw[1]), slice(-shift_hw[1], None):
            shift_mask[:, :, h, w, :] = cnt
            cnt += 1
    shift_mask = cuboid_reorder(shift_mask, (1,) + cuboid_hw, strategy=strategy)
    shift_mask = shift_mask.squeeze(-1).squeeze(0)  # num_cuboids, bH * bW
    shift_mask = (shift_mask.unsqueeze(1) - shift_mask.unsqueeze(2)) == 0  # num_cuboids, bH * bW, bH * bW
    bh_bw = cuboid_hw[0] * cuboid_hw[1]
    attn_mask = shift_mask.reshape((num_cuboids, 1, bh_bw, 1, bh_bw)) * x_mask.reshape((num_cuboids, -1, bh_bw, 1, 1))\
                * mem_mask.reshape(num_cuboids, 1, 1, -1, bh_bw)
    attn_mask = attn_mask.reshape(num_cuboids, x_cuboid_volume, mem_cuboid_volume)
    return attn_mask

class CuboidCrossAttentionLayer(nn.Module):
    """Implements the cuboid cross attention.

    The idea of Cuboid Cross Attention is to extend the idea of cuboid self attention to work for the
    encoder-decoder-type cross attention.

    Assume that there is a memory tensor with shape (T1, H, W, C) and another query tensor with shape (T2, H, W, C),

    Here, we decompose the query tensor and the memory tensor into the same number of cuboids and attend the cuboid in
    the query tensor with the corresponding cuboid in the memory tensor.

    For the height and width axes, we reuse the grid decomposition techniques described in the cuboid self-attention.
    For the temporal axis, the layer supports the "n_temporal" parameter, that controls the number of cuboids we can
    get after cutting the tensors. For example, if the temporal dilation is 2, both the query and
    memory will be decomposed into 2 cuboids along the temporal axis. Like in the Cuboid Self-attention,
    we support "local" and "dilated" decomposition strategy.

    The complexity of the layer is O((T2 / n_t * Bh * Bw) * (T1 / n_t * Bh * Bw) * n_t (H / Bh) (W / Bw)) = O(T2 * T1 / n_t H W Bh Bw)

    """
    def __init__(self,
                 dim,
                 num_heads,
                 n_temporal=1,
                 cuboid_hw=(7, 7),
                 shift_hw=(0, 0),
                 strategy=('d', 'l', 'l'),
                 padding_type='ignore',
                 cross_last_n_frames=None,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 max_temporal_relative=50,
                 norm_layer='layer_norm',
                 use_global_vector=True,
                 separate_global_qkv=False,
                 global_dim_ratio=1,
                 checkpoint_level=1,
                 use_relative_pos=True,
                 attn_linear_init_mode="0",
                 ffn_linear_init_mode="0",
                 norm_init_mode="0",
                 ):
        """

        Parameters
        ----------
        dim
        num_heads
        n_temporal
        cuboid_hw
        shift_hw
            The shift window size as in shifted window attention
        strategy
            The decomposition strategy for the temporal axis, H axis and W axis
        max_temporal_relative
            The maximum temporal relative encoding difference
        cross_last_n_frames
            If provided, only cross attends to the last n frames of `mem`
        use_global_vector
            Whether the memory is coupled with global vectors
        checkpoint_level
            Level of checkpointing:

            0 --> no_checkpointing
            1 --> only checkpoint the FFN
            2 --> checkpoint both FFN and attention
        """
        super(CuboidCrossAttentionLayer, self).__init__()
        # initialization
        self.attn_linear_init_mode = attn_linear_init_mode
        self.ffn_linear_init_mode = ffn_linear_init_mode
        self.norm_init_mode = norm_init_mode

        self.dim = dim
        self.num_heads = num_heads
        self.n_temporal = n_temporal
        assert n_temporal > 0
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        shift_hw = list(shift_hw)
        if strategy[1] == 'd':
            shift_hw[0] = 0
        if strategy[2] == 'd':
            shift_hw[1] = 0
        self.cuboid_hw = cuboid_hw
        self.shift_hw = tuple(shift_hw)
        self.strategy = strategy
        self.padding_type = padding_type
        self.max_temporal_relative = max_temporal_relative
        self.cross_last_n_frames = cross_last_n_frames
        self.use_relative_pos = use_relative_pos
        # global vectors
        self.use_global_vector = use_global_vector
        self.separate_global_qkv = separate_global_qkv
        if global_dim_ratio != 1:
            assert separate_global_qkv == True, \
                f"Setting global_dim_ratio != 1 requires separate_global_qkv == True."
        self.global_dim_ratio = global_dim_ratio

        assert self.padding_type in ['ignore', 'zeros', 'nearest']

        if use_relative_pos:
            # Create relative positional embedding bias table
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * max_temporal_relative - 1) * (2 * cuboid_hw[0] - 1) * (2 * cuboid_hw[1] - 1), num_heads))
            nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

            coords_t = torch.arange(max_temporal_relative)
            coords_h = torch.arange(self.cuboid_hw[0])
            coords_w = torch.arange(self.cuboid_hw[1])
            coords = torch.stack(torch.meshgrid(coords_t, coords_h, coords_w))  # 3, maxT, Bh, Bw

            coords_flatten = torch.flatten(coords, 1)  # 3, maxT*Bh*Bw
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, maxT*Bh*Bw, maxT*Bh*Bw
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # maxT*Bh*Bw, maxT*Bh*Bw, 3
            relative_coords[:, :, 0] += max_temporal_relative - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.cuboid_hw[0] - 1
            relative_coords[:, :, 2] += self.cuboid_hw[1] - 1
            # shape is (cuboid_volume, cuboid_volume)
            relative_position_index = relative_coords[:, :, 0] * (2 * self.cuboid_hw[0] - 1) * (2 * self.cuboid_hw[1] - 1)\
                                      + relative_coords[:, :, 1] * (2 * self.cuboid_hw[1] - 1) + relative_coords[:, :, 2]
            self.register_buffer("relative_position_index", relative_position_index)

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.use_global_vector:
            if self.separate_global_qkv:
                self.l2g_q_net = nn.Linear(dim, dim, bias=qkv_bias)
                self.l2g_global_kv_net = nn.Linear(
                    in_features=global_dim_ratio * dim,
                    out_features=dim * 2,
                    bias=qkv_bias)

        self.norm = get_norm_layer(norm_layer, in_channels=dim)

        self._checkpoint_level = checkpoint_level

        self.reset_parameters()

    def reset_parameters(self):
        apply_initialization(self.q_proj,
                             linear_mode=self.attn_linear_init_mode)
        apply_initialization(self.kv_proj,
                             linear_mode=self.attn_linear_init_mode)
        apply_initialization(self.proj,
                             linear_mode=self.ffn_linear_init_mode)
        apply_initialization(self.norm,
                             norm_mode=self.norm_init_mode)
        if self.use_global_vector:
            if self.separate_global_qkv:
                apply_initialization(self.l2g_q_net,
                                     linear_mode=self.attn_linear_init_mode)
                apply_initialization(self.l2g_global_kv_net,
                                     linear_mode=self.attn_linear_init_mode)

    def forward(self, x, mem, mem_global_vectors=None):
        """Calculate the forward

        Along the temporal axis, we pad the mem tensor from the left and the x tensor from the right so that the
        relative position encoding can be calculated correctly. For example:

        mem: 0, 1, 2, 3, 4
        x:   0, 1, 2, 3, 4, 5

        n_temporal = 1
        mem: 0, 1, 2, 3, 4   x: 0, 1, 2, 3, 4, 5

        n_temporal = 2
        mem: pad, 1, 3       x: 0, 2, 4
        mem: 0, 2, 4         x: 1, 3, 5

        n_temporal = 3
        mem: pad, 2          dec: 0, 3
        mem: 0,   3          dec: 1, 4
        mem: 1,   4          dec: 2, 5

        Parameters
        ----------
        x
            The input of the layer. It will have shape (B, T, H, W, C)
        mem
            The memory. It will have shape (B, T_mem, H, W, C)
        mem_global_vectors
            The global vectors from the memory. It will have shape (B, N, C)

        Returns
        -------
        out
            Output tensor should have shape (B, T, H, W, C_out)
        """
        if self.cross_last_n_frames is not None:
            cross_last_n_frames = int(min(self.cross_last_n_frames, mem.shape[1]))
            mem = mem[:, -cross_last_n_frames:, ...]
        if self.use_global_vector:
            _, num_global, _ = mem_global_vectors.shape
        x = self.norm(x)
        B, T_x, H, W, C_in = x.shape
        B_mem, T_mem, H_mem, W_mem, C_mem = mem.shape
        assert T_x < self.max_temporal_relative and T_mem < self.max_temporal_relative
        cuboid_hw = self.cuboid_hw
        n_temporal = self.n_temporal
        shift_hw = self.shift_hw
        assert B_mem == B and H == H_mem and W == W_mem and C_in == C_mem,\
            f'Shape of memory and the input tensor does not match. x.shape={x.shape}, mem.shape={mem.shape}'
        pad_t_mem = (n_temporal - T_mem % n_temporal) % n_temporal
        pad_t_x = (n_temporal - T_x % n_temporal) % n_temporal
        pad_h = (cuboid_hw[0] - H % cuboid_hw[0]) % cuboid_hw[0]
        pad_w = (cuboid_hw[1] - W % cuboid_hw[1]) % cuboid_hw[1]

        # Step-1: Pad the memory and x
        mem = _generalize_padding(mem, pad_t_mem, pad_h, pad_w, self.padding_type, t_pad_left=True)
        x = _generalize_padding(x, pad_t_x, pad_h, pad_w, self.padding_type, t_pad_left=False)

        # Step-2: Shift the tensor based on shift window attention.
        if any(i > 0 for i in shift_hw):
            shifted_x = torch.roll(x, shifts=(-shift_hw[0], -shift_hw[1]), dims=(2, 3))
            shifted_mem = torch.roll(mem, shifts=(-shift_hw[0], -shift_hw[1]), dims=(2, 3))
        else:
            shifted_x = x
            shifted_mem = mem

        # Step-3: Reorder the tensors
        mem_cuboid_size = (mem.shape[1] // n_temporal,) + cuboid_hw
        x_cuboid_size = (x.shape[1] // n_temporal,) + cuboid_hw

        # Mem shape is (B, num_cuboids, mem_cuboid_volume, C), x shape is (B, num_cuboids, x_cuboid_volume, C)
        reordered_mem = cuboid_reorder(shifted_mem, cuboid_size=mem_cuboid_size, strategy=self.strategy)
        reordered_x = cuboid_reorder(shifted_x, cuboid_size=x_cuboid_size, strategy=self.strategy)
        _, num_cuboids_mem, mem_cuboid_volume, _ = reordered_mem.shape
        _, num_cuboids, x_cuboid_volume, _ = reordered_x.shape
        assert num_cuboids_mem == num_cuboids, f'Number of cuboids do not match. num_cuboids={num_cuboids},' \
                                               f' num_cuboids_mem={num_cuboids_mem}'

        # Step-4: Perform self-attention
        # (num_cuboids, x_cuboid_volume, mem_cuboid_volume)
        attn_mask = compute_cuboid_cross_attention_mask(T_x, T_mem, H, W, n_temporal, cuboid_hw, shift_hw,
                                                        strategy=self.strategy,
                                                        padding_type=self.padding_type,
                                                        device=x.device)
        head_C = C_in // self.num_heads

        # (2, B, num_heads, num_cuboids, mem_cuboid_volume, head_C)
        kv = self.kv_proj(reordered_mem).reshape(B, num_cuboids, mem_cuboid_volume, 2, self.num_heads, head_C).permute(3, 0, 4, 1, 2, 5)
        k, v = kv[0], kv[1]  # Each has shape (B, num_heads, num_cuboids, mem_cuboid_volume, head_C)
        q = self.q_proj(reordered_x).reshape(B, num_cuboids, x_cuboid_volume, self.num_heads, head_C).permute(0, 3, 1, 2, 4)  # Shape (B, num_heads, num_cuboids, x_cuboids_volume, head_C)
        q = q * self.scale
        attn_score = q @ k.transpose(-2, -1)  # Shape (B, num_heads, num_cuboids, x_cuboids_volume, mem_cuboid_volume)

        if self.use_relative_pos:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index[:x_cuboid_volume, :mem_cuboid_volume].reshape(-1)].reshape(
                x_cuboid_volume, mem_cuboid_volume, -1)  # (cuboid_volume, cuboid_volume, num_head)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(1)  # num_heads, 1, x_cuboids_volume, mem_cuboid_volume
            attn_score = attn_score + relative_position_bias  # Shape (B, num_heads, num_cuboids, x_cuboids_volume, mem_cuboid_volume)

        if self.use_global_vector:
            if self.separate_global_qkv:
                l2g_q = self.l2g_q_net(reordered_x) \
                    .reshape(B, num_cuboids, x_cuboid_volume, self.num_heads, head_C) \
                    .permute(0, 3, 1, 2, 4)  # (B, num_heads, num_cuboids, cuboid_volume, head_C)
                l2g_q = l2g_q * self.scale
                l2g_global_kv = self.l2g_global_kv_net(mem_global_vectors) \
                    .reshape(B, 1, num_global, 2, self.num_heads, head_C) \
                    .permute(3, 0, 4, 1, 2, 5)  # Shape (2, B, num_heads, 1, N, head_C)
                l2g_global_k, l2g_global_v = l2g_global_kv[0], l2g_global_kv[1]
            else:
                kv_global = self.kv_proj(mem_global_vectors).reshape(B, 1, num_global, 2, self.num_heads, head_C).permute(3, 0, 4, 1, 2, 5)
                l2g_global_k, l2g_global_v = kv_global[0], kv_global[1]  # Shape (B, num_heads, 1, num_global, head_C)
                l2g_q = q
            l2g_attn_score = l2g_q @ l2g_global_k.transpose(-2, -1)  # Shape (B, num_heads, num_cuboids, x_cuboid_volume, num_global)
            attn_score_l2l_l2g = torch.cat((attn_score, l2g_attn_score),
                                           dim=-1)
            attn_mask_l2l_l2g = F.pad(attn_mask, (0, num_global), "constant", 1)  # Shape (num_cuboids, x_cuboid_volume, mem_cuboid_volume + num_global)
            v_l_g = torch.cat((v, l2g_global_v.expand(B, self.num_heads, num_cuboids, num_global, head_C)),
                              dim=3)  # Shape (B, num_heads, num_cuboids, mem_cuboid_volume + num_global, head_C)
            # local to local and global attention
            attn_score_l2l_l2g = masked_softmax(attn_score_l2l_l2g, mask=attn_mask_l2l_l2g)
            attn_score_l2l_l2g = self.attn_drop(attn_score_l2l_l2g)  # Shape (B, num_heads, num_cuboids, x_cuboid_volume, mem_cuboid_volume + K))
            reordered_x = (attn_score_l2l_l2g @ v_l_g).permute(0, 2, 3, 1, 4) \
                .reshape(B, num_cuboids, x_cuboid_volume, self.dim)
        else:
            attn_score = masked_softmax(attn_score, mask=attn_mask)
            attn_score = self.attn_drop(attn_score)  # Shape (B, num_heads, num_cuboids, x_cuboid_volume, mem_cuboid_volume)
            reordered_x = (attn_score @ v).permute(0, 2, 3, 1, 4).reshape(B, num_cuboids, x_cuboid_volume, self.dim)
        reordered_x = self.proj_drop(self.proj(reordered_x))
        # Step-5: Shift back and slice
        shifted_x = cuboid_reorder_reverse(reordered_x, cuboid_size=x_cuboid_size, strategy=self.strategy,
                                           orig_data_shape=(x.shape[1], x.shape[2], x.shape[3]))
        if any(i > 0 for i in shift_hw):
            x = torch.roll(shifted_x, shifts=(shift_hw[0], shift_hw[1]), dims=(2, 3))
        else:
            x = shifted_x
        x = _generalize_unpadding(x, pad_t=pad_t_x, pad_h=pad_h, pad_w=pad_w, padding_type=self.padding_type)
        return x

class DownSampling3D(nn.Module):
    """The 3D down-sampling layer.

    3d_interp_2d:
        x --> conv3d_3X3X3 (mid_dim) + leaky_relu --> downsample (bilinear) --> conv2d_3x3
    2d_interp_2d:
        x --> conv2d_3x3 (mid_dim) + leaky_relu --> downsample (bilinear) --> conv2d_3x3

    We add an additional conv layer before the

    For any options, if the target_size is the same as the input size, we will skip the bilinear downsampling layer.
    """
    def __init__(self, original_size, target_size, in_channels, out_dim, mid_dim=16, act_type='leaky',
                 arch_type='2d_interp_2d'):
        """

        Parameters
        ----------
        original_size
            The original size of the tensor. It will be a tuple/list that contains T, H, W
        target_size
            Will be a tuple/list that contains T_new, H_new, W_new
        in_channels
            The input channels
        out_dim
            The output dimension of the layer
        mid_dim
            Dimension of the intermediate projection layer
        act_type
            Type of the activation
        arch_type
            Type of the layer.
        """
        super(DownSampling3D, self).__init__()
        self.arch_type = arch_type
        self.original_size = original_size
        self.target_size = target_size
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        if self.arch_type == '3d_interp_2d':
            self.inter_conv = nn.Conv3d(in_channels=in_channels, out_channels=mid_dim, kernel_size=(3, 3, 3),
                                        padding=(1, 1, 1))
            self.act = get_activation(act_type)
        elif self.arch_type == '2d_interp_2d':
            self.inter_conv = nn.Conv2d(in_channels=in_channels, out_channels=mid_dim, kernel_size=(3, 3),
                                        padding=(1, 1))
            self.act = get_activation(act_type)
        else:
            raise NotImplementedError
        self.conv = nn.Conv2d(in_channels=mid_dim, out_channels=out_dim, kernel_size=(3, 3), padding=(1, 1))
        self.init_weights()

    def init_weights(self):
        for m in self.children():
            apply_initialization(m)

    def forward(self, x):
        """

        Parameters
        ----------
        x
            Shape (N, T, H, W, C)

        Returns
        -------
        out
            Shape (N, T_new, H_new, W_new, C_out)
        """
        B, T, H, W, C_in = x.shape
        if self.arch_type == '3d_interp_2d':
            x = self.act(self.inter_conv(x.permute(0, 4, 1, 2, 3)))  # Shape(B, mid_dim, T, H, W)
            if self.original_size[0] == self.target_size[0]:
                # Use 2D interpolation
                x = F.interpolate(x.permute(0, 2, 1, 3, 4).reshape(B * T, self.mid_dim, H, W), size=self.target_size[1:])  # Shape (B * T_new, mid_dim, H_new, W_new)
            else:
                # Use 3D interpolation
                x = F.interpolate(x, size=self.target_size)  # Shape (B, mid_dim, T_new, H_new, W_new)
                x = x.permute(0, 2, 1, 3, 4).reshape(B * self.target_size[0], self.mid_dim,
                                                     self.target_size[1], self.target_size[2])
        elif self.arch_type == '2d_interp_2d':
            x = self.act(self.inter_conv(x.permute(0, 1, 4, 2, 3).reshape(B * T, C_in, H, W)))  # (B * T, mid_dim, H, W)

            if self.original_size[0] == self.target_size[0]:
                # Use 2D interpolation
                x = F.interpolate(x, size=self.target_size[1:])  # Shape (B * T_new, mid_dim, H_new, W_new)
            else:
                # Use 3D interpolation
                x = F.interpolate(x.reshape(B, T, C_in, H, W).permute(0, 2, 1, 3, 4), size=self.target_size)  # Shape (B, mid_dim, T_new, H_new, W_new)
                x = x.permute(0, 2, 1, 3, 4).reshape(B * self.target_size[0], self.mid_dim,
                                                     self.target_size[1], self.target_size[2])
        else:
            raise NotImplementedError
        x = self.conv(x)  # Shape (B * T_new, out_dim, H_new, W_new)
        x = x.reshape(B, self.target_size[0], self.out_dim, self.target_size[1], self.target_size[2])\
             .permute(0, 2, 1, 3, 4)
        return x

class CuboidTransformerEncoder(nn.Module):
    """Encoder of the CuboidTransformer

    x --> attn_block --> patch_merge --> attn_block --> patch_merge --> ... --> out

    """
    def __init__(self,
                 input_shape,
                 base_units=128,
                 block_units=None,
                 scale_alpha=1.0,
                 depth=[4, 4, 4],
                 downsample=2,
                 downsample_type='patch_merge',
                 block_attn_patterns=None,
                 block_cuboid_size=[(4, 4, 4),
                                    (4, 4, 4)],
                 block_strategy=[('l', 'l', 'l'),
                                 ('d', 'd', 'd')],
                 block_shift_size=[(0, 0, 0),
                                   (0, 0, 0)],
                 num_heads=4,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 ffn_drop=0.0,
                 activation="leaky",
                 ffn_activation='leaky',
                 gated_ffn=False,
                 norm_layer='layer_norm',
                 use_inter_ffn=True,
                 padding_type='ignore',
                 checkpoint_level=True,
                 use_relative_pos=True,
                 self_attn_use_final_proj=True,
                 # global vectors
                 use_global_vector=False,
                 use_global_vector_ffn=True,
                 use_global_self_attn=False,
                 separate_global_qkv=False,
                 global_dim_ratio=1,
                 # initialization
                 attn_linear_init_mode="0",
                 ffn_linear_init_mode="0",
                 conv_init_mode="0",
                 down_linear_init_mode="0",
                 norm_init_mode="0",
                 ):
        """

        Parameters
        ----------
        input_shape
            The shape of the input. Contains T, H, W, C
        initial_data_thw
            The shape of the first layer
        base_units
            The number of units
        scale_alpha
            We scale up the channels based on the formula:
            - round_to(base_units * max(downsample_scale) ** units_alpha, 4)
        depth
            The number of layers for each block
        downsample
            The downsample ratio
        downsample_type
            Type of the downsampling layer
        block_attn_patterns
            Attention pattern for the cuboid attention for each block.
        block_cuboid_size
            A list of cuboid size parameters
        block_strategy
            A list of cuboid strategies
        block_shift_size
            A list of shift sizes
        num_global
            The number of global vectors
        num_heads
            The number of heads.
        attn_drop
        proj_drop
        ffn_drop
        gated_ffn
            Whether to enable gated ffn or not
        norm_layer
            The normalization layer
        use_inter_ffn
            Whether to use intermediate FFN
        padding_type
        """
        super(CuboidTransformerEncoder, self).__init__()
        # initialization mode
        self.attn_linear_init_mode = attn_linear_init_mode
        self.ffn_linear_init_mode = ffn_linear_init_mode
        self.conv_init_mode = conv_init_mode
        self.down_linear_init_mode = down_linear_init_mode
        self.norm_init_mode = norm_init_mode

        self.input_shape = input_shape
        self.depth = depth
        self.num_blocks = len(depth)
        self.base_units = base_units
        self.scale_alpha = scale_alpha
        if not isinstance(downsample, (tuple, list)):
            downsample = (1, downsample, downsample)
        self.downsample = downsample
        self.downsample_type = downsample_type
        self.num_heads = num_heads
        self.use_global_vector = use_global_vector
        self.checkpoint_level = checkpoint_level
        if block_units is None:
            block_units = [round_to(base_units * int((max(downsample) ** scale_alpha) ** i), 4)
                           for i in range(self.num_blocks)]
        else:
            assert len(block_units) == self.num_blocks and block_units[0] == base_units
        self.block_units = block_units

        if self.num_blocks > 1:
            if downsample_type == 'patch_merge':
                self.down_layers = nn.ModuleList(
                    [PatchMerging3D(dim=self.block_units[i],
                                    downsample=downsample,
                                    # downsample=(1, 1, 1),
                                    padding_type=padding_type,
                                    out_dim=self.block_units[i + 1],
                                    linear_init_mode=down_linear_init_mode,
                                    norm_init_mode=norm_init_mode)
                     for i in range(self.num_blocks - 1)])
            else:
                raise NotImplementedError
            if self.use_global_vector:
                self.down_layer_global_proj = nn.ModuleList(
                    [nn.Linear(in_features=global_dim_ratio*self.block_units[i],
                               out_features=global_dim_ratio*self.block_units[i + 1])
                     for i in range(self.num_blocks - 1)])

        if block_attn_patterns is not None:
            mem_shapes = self.get_mem_shapes()
            if isinstance(block_attn_patterns, (tuple, list)):
                assert len(block_attn_patterns) == self.num_blocks
            else:
                block_attn_patterns = [block_attn_patterns for _ in range(self.num_blocks)]
            block_cuboid_size = []
            block_strategy = []
            block_shift_size = []
            for idx, key in enumerate(block_attn_patterns):
                func = CuboidSelfAttentionPatterns.get(key)
                cuboid_size, strategy, shift_size = func(mem_shapes[idx])
                block_cuboid_size.append(cuboid_size)
                block_strategy.append(strategy)
                block_shift_size.append(shift_size)
        else:
            if not isinstance(block_cuboid_size[0][0], (list, tuple)):
                block_cuboid_size = [block_cuboid_size for _ in range(self.num_blocks)]
            else:
                assert len(block_cuboid_size) == self.num_blocks,\
                    f'Incorrect input format! Received block_cuboid_size={block_cuboid_size}'

            if not isinstance(block_strategy[0][0], (list, tuple)):
                block_strategy = [block_strategy for _ in range(self.num_blocks)]
            else:
                assert len(block_strategy) == self.num_blocks,\
                    f'Incorrect input format! Received block_strategy={block_strategy}'

            if not isinstance(block_shift_size[0][0], (list, tuple)):
                block_shift_size = [block_shift_size for _ in range(self.num_blocks)]
            else:
                assert len(block_shift_size) == self.num_blocks,\
                    f'Incorrect input format! Received block_shift_size={block_shift_size}'
        self.block_cuboid_size = block_cuboid_size
        self.block_strategy = block_strategy
        self.block_shift_size = block_shift_size

        self.blocks = nn.ModuleList([nn.Sequential(
            *[StackCuboidSelfAttentionBlock(
                dim=self.block_units[i],
                num_heads=num_heads,
                block_cuboid_size=block_cuboid_size[i],
                block_strategy=block_strategy[i],
                block_shift_size=block_shift_size[i],
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                ffn_drop=ffn_drop,
                activation=ffn_activation,
                gated_ffn=gated_ffn,
                norm_layer=norm_layer,
                use_inter_ffn=use_inter_ffn,
                padding_type=padding_type,
                use_global_vector=use_global_vector,
                use_global_vector_ffn=use_global_vector_ffn,
                use_global_self_attn=use_global_self_attn,
                separate_global_qkv=separate_global_qkv,
                global_dim_ratio=global_dim_ratio,
                checkpoint_level=checkpoint_level,
                use_relative_pos=use_relative_pos,
                use_final_proj=self_attn_use_final_proj,
                # initialization
                attn_linear_init_mode=attn_linear_init_mode,
                ffn_linear_init_mode=ffn_linear_init_mode,
                norm_init_mode=norm_init_mode,
            ) for _ in range(depth[i])])
            for i in range(self.num_blocks)])
        self.reset_parameters()

    def reset_parameters(self):
        if self.num_blocks > 1:
            for m in self.down_layers:
                m.reset_parameters()
            if self.use_global_vector:
                apply_initialization(self.down_layer_global_proj,
                                     linear_mode=self.down_linear_init_mode)
        for ms in self.blocks:
            for m in ms:
                m.reset_parameters()

    def get_mem_shapes(self):
        """Get the shape of the output memory based on the input shape. This can be used for constructing the decoder.

        Returns
        -------
        mem_shapes
            A list of shapes of the output memory
        """

        if self.num_blocks == 1:
            return [self.input_shape]
        else:
            mem_shapes = [self.input_shape]
            curr_shape = self.input_shape
            for down_layer in self.down_layers:
                curr_shape = down_layer.get_out_shape(curr_shape)
                mem_shapes.append(curr_shape)
            return mem_shapes

    def forward(self, x, global_vectors=None):
        """

        Parameters
        ----------
        x
            Shape (B, T, H, W, C)

        Returns
        -------
        out
            A list of tensors from the bottom layer to the top layer of the encoder. For example, it can have shape
            - (B, T, H, W, C1)
            - (B, T, H // 2, W // 2, 2 * C1)
            - (B, T, H // 4, W // 4, 4 * C1)
            ...
        global_mem_out
            Optional
        """
        B, T, H, W, C_in = x.shape
        assert (T, H, W, C_in) == self.input_shape

        if self.use_global_vector:
            out = []
            global_mem_out = []
            for i in range(self.num_blocks):
                for l in self.blocks[i]:
                    x, global_vectors = l(x, global_vectors)
                out.append(x)
                global_mem_out.append(global_vectors)
                if self.num_blocks > 1 and i < self.num_blocks - 1:
                    x = self.down_layers[i](x)
                    global_vectors = self.down_layer_global_proj[i](global_vectors)
            return out, global_mem_out
        else:
            out = []
            for i in range(self.num_blocks):
                x = self.blocks[i](x)
                out.append(x)
                if self.num_blocks > 1 and i < self.num_blocks - 1:
                    x = self.down_layers[i](x)
            return out

class StackCuboidCrossAttentionBlock(nn.Module):
    """A stack of cuboid cross attention layers.

    The advantage of cuboid attention is that we can combine cuboid attention building blocks with different
    hyper-parameters to mimic a broad range of space-time correlation patterns.

    - "use_inter_ffn" is True
        x, mem --> attn1 -----+-------> ffn1 ---+---> attn2 --> ... --> ffn_k --> out
           |             ^    |             ^
           |             |    |             |
           |-------------|----|-------------|
    - "use_inter_ffn" is False
        x, mem --> attn1 -----+------> attn2 --> ... attnk --+----> ffnk ---+---> out, mem
           |             ^    |            ^             ^  |           ^
           |             |    |            |             |  |           |
           |-------------|----|------------|-- ----------|--|-----------|
    """
    def __init__(self,
                 dim,
                 num_heads,
                 block_cuboid_hw=[(4, 4), (4, 4)],
                 block_shift_hw=[(0, 0), (2, 2)],
                 block_n_temporal=[1, 2],
                 block_strategy=[('d', 'd', 'd'),
                                 ('l', 'l', 'l')],
                 padding_type='ignore',
                 cross_last_n_frames=None,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 ffn_drop=0.0,
                 activation='leaky',
                 gated_ffn=False,
                 norm_layer='layer_norm',
                 use_inter_ffn=True,
                 max_temporal_relative=50,
                 checkpoint_level=1,
                 use_relative_pos=True,
                 # global vectors
                 use_global_vector=False,
                 separate_global_qkv=False,
                 global_dim_ratio=1,
                 # initialization
                 attn_linear_init_mode="0",
                 ffn_linear_init_mode="0",
                 norm_init_mode="0",
                 ):
        super(StackCuboidCrossAttentionBlock, self).__init__()
        # initialization
        self.attn_linear_init_mode = attn_linear_init_mode
        self.ffn_linear_init_mode = ffn_linear_init_mode
        self.norm_init_mode = norm_init_mode

        assert len(block_cuboid_hw[0]) > 0 and len(block_shift_hw) > 0 and len(block_strategy) > 0,\
            f'Incorrect format.' \
            f' block_cuboid_hw={block_cuboid_hw}, block_shift_hw={block_shift_hw}, block_strategy={block_strategy}'
        assert len(block_cuboid_hw) == len(block_shift_hw) == len(block_strategy)
        self.num_attn = len(block_cuboid_hw)
        self.checkpoint_level = checkpoint_level
        self.use_inter_ffn = use_inter_ffn
        self.use_global_vector = use_global_vector
        if self.use_inter_ffn:
            self.ffn_l = nn.ModuleList(
                [PositionwiseFFN(
                    units=dim,
                    hidden_size=4 * dim,
                    activation_dropout=ffn_drop,
                    dropout=ffn_drop,
                    gated_proj=gated_ffn,
                    activation=activation,
                    normalization=norm_layer,
                    pre_norm=True,
                    linear_init_mode=ffn_linear_init_mode,
                    norm_init_mode=norm_init_mode,)
                    for _ in range(self.num_attn)])
        else:
            self.ffn_l = nn.ModuleList(
                [PositionwiseFFN(
                    units=dim,
                    hidden_size=4 * dim,
                    activation_dropout=ffn_drop,
                    dropout=ffn_drop,
                    gated_proj=gated_ffn,
                    activation=activation,
                    normalization=norm_layer,
                    pre_norm=True,
                    linear_init_mode=ffn_linear_init_mode,
                    norm_init_mode=norm_init_mode,)])
        self.attn_l = nn.ModuleList(
            [CuboidCrossAttentionLayer(
                dim=dim,
                num_heads=num_heads,
                cuboid_hw=ele_cuboid_hw,
                shift_hw=ele_shift_hw,
                strategy=ele_strategy,
                n_temporal=ele_n_temporal,
                cross_last_n_frames=cross_last_n_frames,
                padding_type=padding_type,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                norm_layer=norm_layer,
                max_temporal_relative=max_temporal_relative,
                use_global_vector=use_global_vector,
                separate_global_qkv=separate_global_qkv,
                global_dim_ratio=global_dim_ratio,
                checkpoint_level=checkpoint_level,
                use_relative_pos=use_relative_pos,
                attn_linear_init_mode=attn_linear_init_mode,
                ffn_linear_init_mode=ffn_linear_init_mode,
                norm_init_mode=norm_init_mode,)
                for ele_cuboid_hw, ele_shift_hw, ele_strategy, ele_n_temporal
                in zip(block_cuboid_hw, block_shift_hw, block_strategy, block_n_temporal)])

    def reset_parameters(self):
        for m in self.ffn_l:
            m.reset_parameters()
        for m in self.attn_l:
            m.reset_parameters()

    def forward(self, x, mem, mem_global_vector=None):
        """

        Parameters
        ----------
        x
            Shape (B, T_x, H, W, C)
        mem
            Shape (B, T_mem, H, W, C)
        mem_global_vector
            Shape (B, N_global, C)

        Returns
        -------
        out
            Shape (B, T_x, H, W, C_out)
        """
        if self.use_inter_ffn:
            for attn, ffn in zip(self.attn_l, self.ffn_l):
                if self.checkpoint_level >= 2 and self.training:
                    x = x + checkpoint.checkpoint(attn, x, mem, mem_global_vector)
                else:
                    x = x + attn(x, mem, mem_global_vector)
                if self.checkpoint_level >= 1 and self.training:
                    x = checkpoint.checkpoint(ffn, x)
                else:
                    x = ffn(x)
            return x
        else:
            for attn in self.attn_l:
                if self.checkpoint_level >= 2 and self.training:
                    x = x + checkpoint.checkpoint(attn, x, mem, mem_global_vector)
                else:
                    x = x + attn(x, mem, mem_global_vector)
            if self.checkpoint_level >= 1 and self.training:
                x = checkpoint.checkpoint(self.ffn_l[0], x)
            else:
                x = self.ffn_l[0](x)
        return x

class CuboidTransformerDecoder(nn.Module):
    """Decoder of the CuboidTransformer.

    For each block, we first apply the StackCuboidSelfAttention and then apply the StackCuboidCrossAttention

    Repeat the following structure K times

        x --> StackCuboidSelfAttention --> |
                                           |----> StackCuboidCrossAttention (If used) --> out
                                   mem --> |

    """
    def __init__(self,
                 target_temporal_length,
                 mem_shapes,
                 cross_start=0,
                 depth=[2, 2],
                 upsample_type="upsample",
                 upsample_kernel_size=3,
                 block_self_attn_patterns=None,
                 block_self_cuboid_size=[(4, 4, 4), (4, 4, 4)],
                 block_self_cuboid_strategy=[('l', 'l', 'l'), ('d', 'd', 'd')],
                 block_self_shift_size=[(1, 1, 1), (0, 0, 0)],
                 block_cross_attn_patterns=None,
                 block_cross_cuboid_hw=[(4, 4), (4, 4)],
                 block_cross_cuboid_strategy=[('l', 'l', 'l'), ('d', 'l', 'l')],
                 block_cross_shift_hw=[(0, 0), (0, 0)],
                 block_cross_n_temporal=[1, 2],
                 cross_last_n_frames=None,
                 num_heads=4,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 ffn_drop=0.0,
                 ffn_activation='leaky',
                 gated_ffn=False,
                 norm_layer='layer_norm',
                 use_inter_ffn=False,
                 hierarchical_pos_embed=False,
                 pos_embed_type='t+hw',
                 max_temporal_relative=50,
                 padding_type='ignore',
                 checkpoint_level=True,
                 use_relative_pos=True,
                 self_attn_use_final_proj=True,
                 use_first_self_attn=False,
                 # global vectors
                 use_self_global=False,
                 self_update_global=True,
                 use_cross_global=False,
                 use_global_vector_ffn=True,
                 use_global_self_attn=False,
                 separate_global_qkv=False,
                 global_dim_ratio=1,
                 # initialization
                 attn_linear_init_mode="0",
                 ffn_linear_init_mode="0",
                 conv_init_mode="0",
                 up_linear_init_mode="0",
                 norm_init_mode="0",
                 ):
        """

        Parameters
        ----------
        target_temporal_length
        mem_shapes
        cross_start
            The block to start cross attention
        depth
            Depth of each block
        upsample_type
            The type of the upsampling layers
        upsample_kernel_size
        block_self_attn_patterns
            Pattern of the block self attentions
        block_self_cuboid_size
        block_self_cuboid_strategy
        block_self_shift_size
        block_cross_attn_patterns
        block_cross_cuboid_hw
        block_cross_cuboid_strategy
        block_cross_shift_hw
        block_cross_n_temporal
        num_heads
        attn_drop
        proj_drop
        ffn_drop
        ffn_activation
        gated_ffn
        norm_layer
        use_inter_ffn
        hierarchical_pos_embed
            Whether to add pos embedding for each hierarchy.
        max_temporal_relative
        padding_type
        checkpoint_level
        """
        super(CuboidTransformerDecoder, self).__init__()
        # initialization mode
        self.attn_linear_init_mode = attn_linear_init_mode
        self.ffn_linear_init_mode = ffn_linear_init_mode
        self.conv_init_mode = conv_init_mode
        self.up_linear_init_mode = up_linear_init_mode
        self.norm_init_mode = norm_init_mode

        assert len(depth) == len(mem_shapes)
        self.target_temporal_length = target_temporal_length
        self.num_blocks = len(mem_shapes)
        self.cross_start = cross_start
        self.mem_shapes = mem_shapes
        self.depth = depth
        self.upsample_type = upsample_type
        self.hierarchical_pos_embed = hierarchical_pos_embed
        self.checkpoint_level = checkpoint_level
        self.use_self_global = use_self_global
        self.self_update_global = self_update_global
        self.use_cross_global = use_cross_global
        self.use_global_vector_ffn = use_global_vector_ffn
        self.use_first_self_attn = use_first_self_attn
        if block_self_attn_patterns is not None:
            if isinstance(block_self_attn_patterns, (tuple, list)):
                assert len(block_self_attn_patterns) == self.num_blocks
            else:
                block_self_attn_patterns = [block_self_attn_patterns for _ in range(self.num_blocks)]
            block_self_cuboid_size = []
            block_self_cuboid_strategy = []
            block_self_shift_size = []
            for idx, key in enumerate(block_self_attn_patterns):
                func = CuboidSelfAttentionPatterns.get(key)
                cuboid_size, strategy, shift_size = func(mem_shapes[idx])
                block_self_cuboid_size.append(cuboid_size)
                block_self_cuboid_strategy.append(strategy)
                block_self_shift_size.append(shift_size)
        else:
            if not isinstance(block_self_cuboid_size[0][0], (list, tuple)):
                block_self_cuboid_size = [block_self_cuboid_size for _ in range(self.num_blocks)]
            else:
                assert len(block_self_cuboid_size) == self.num_blocks,\
                    f'Incorrect input format! Received block_self_cuboid_size={block_self_cuboid_size}'

            if not isinstance(block_self_cuboid_strategy[0][0], (list, tuple)):
                block_self_cuboid_strategy = [block_self_cuboid_strategy for _ in range(self.num_blocks)]
            else:
                assert len(block_self_cuboid_strategy) == self.num_blocks,\
                    f'Incorrect input format! Received block_self_cuboid_strategy={block_self_cuboid_strategy}'

            if not isinstance(block_self_shift_size[0][0], (list, tuple)):
                block_self_shift_size = [block_self_shift_size for _ in range(self.num_blocks)]
            else:
                assert len(block_self_shift_size) == self.num_blocks,\
                    f'Incorrect input format! Received block_self_shift_size={block_self_shift_size}'
        self_blocks = []
        for i in range(self.num_blocks):
            if not self.use_first_self_attn and i == self.num_blocks - 1:
                # For the top block, we won't use an additional self attention layer.
                ele_depth = depth[i] - 1
            else:
                ele_depth = depth[i]
            stack_cuboid_blocks =\
                [StackCuboidSelfAttentionBlock(
                    dim=self.mem_shapes[i][-1],
                    num_heads=num_heads,
                    block_cuboid_size=block_self_cuboid_size[i],
                    block_strategy=block_self_cuboid_strategy[i],
                    block_shift_size=block_self_shift_size[i],
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    ffn_drop=ffn_drop,
                    activation=ffn_activation,
                    gated_ffn=gated_ffn,
                    norm_layer=norm_layer,
                    use_inter_ffn=use_inter_ffn,
                    padding_type=padding_type,
                    use_global_vector=use_self_global,
                    use_global_vector_ffn=use_global_vector_ffn,
                    use_global_self_attn=use_global_self_attn,
                    separate_global_qkv=separate_global_qkv,
                    global_dim_ratio=global_dim_ratio,
                    checkpoint_level=checkpoint_level,
                    use_relative_pos=use_relative_pos,
                    use_final_proj=self_attn_use_final_proj,
                    # initialization
                    attn_linear_init_mode=attn_linear_init_mode,
                    ffn_linear_init_mode=ffn_linear_init_mode,
                    norm_init_mode=norm_init_mode,
                ) for _ in range(ele_depth)]
            self_blocks.append(nn.ModuleList(stack_cuboid_blocks))
        self.self_blocks = nn.ModuleList(self_blocks)

        if block_cross_attn_patterns is not None:
            if isinstance(block_cross_attn_patterns, (tuple, list)):
                assert len(block_cross_attn_patterns) == self.num_blocks
            else:
                block_cross_attn_patterns = [block_cross_attn_patterns for _ in range(self.num_blocks)]

            block_cross_cuboid_hw = []
            block_cross_cuboid_strategy = []
            block_cross_shift_hw = []
            block_cross_n_temporal = []
            for idx, key in enumerate(block_cross_attn_patterns):
                if key == "last_frame_dst":
                    cuboid_hw = None
                    shift_hw = None
                    strategy = None
                    n_temporal = None
                else:
                    func = CuboidCrossAttentionPatterns.get(key)
                    cuboid_hw, shift_hw, strategy, n_temporal = func(mem_shapes[idx])
                block_cross_cuboid_hw.append(cuboid_hw)
                block_cross_cuboid_strategy.append(strategy)
                block_cross_shift_hw.append(shift_hw)
                block_cross_n_temporal.append(n_temporal)
        else:
            if not isinstance(block_cross_cuboid_hw[0][0], (list, tuple)):
                block_cross_cuboid_hw = [block_cross_cuboid_hw for _ in range(self.num_blocks)]
            else:
                assert len(block_cross_cuboid_hw) == self.num_blocks, \
                    f'Incorrect input format! Received block_cross_cuboid_hw={block_cross_cuboid_hw}'

            if not isinstance(block_cross_cuboid_strategy[0][0], (list, tuple)):
                block_cross_cuboid_strategy = [block_cross_cuboid_strategy for _ in range(self.num_blocks)]
            else:
                assert len(block_cross_cuboid_strategy) == self.num_blocks, \
                    f'Incorrect input format! Received block_cross_cuboid_strategy={block_cross_cuboid_strategy}'

            if not isinstance(block_cross_shift_hw[0][0], (list, tuple)):
                block_cross_shift_hw = [block_cross_shift_hw for _ in range(self.num_blocks)]
            else:
                assert len(block_cross_shift_hw) == self.num_blocks, \
                    f'Incorrect input format! Received block_cross_shift_hw={block_cross_shift_hw}'
            if not isinstance(block_cross_n_temporal[0], (list, tuple)):
                block_cross_n_temporal = [block_cross_n_temporal for _ in range(self.num_blocks)]
            else:
                assert len(block_cross_n_temporal) == self.num_blocks, \
                    f'Incorrect input format! Received block_cross_n_temporal={block_cross_n_temporal}'
        self.cross_blocks = nn.ModuleList()
        for i in range(self.cross_start, self.num_blocks):
            cross_block = nn.ModuleList(
                [StackCuboidCrossAttentionBlock(
                    dim=self.mem_shapes[i][-1],
                    num_heads=num_heads,
                    block_cuboid_hw=block_cross_cuboid_hw[i],
                    block_strategy=block_cross_cuboid_strategy[i],
                    block_shift_hw=block_cross_shift_hw[i],
                    block_n_temporal=block_cross_n_temporal[i],
                    cross_last_n_frames=cross_last_n_frames,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    ffn_drop=ffn_drop,
                    gated_ffn=gated_ffn,
                    norm_layer=norm_layer,
                    use_inter_ffn=use_inter_ffn,
                    activation=ffn_activation,
                    max_temporal_relative=max_temporal_relative,
                    padding_type=padding_type,
                    use_global_vector=use_cross_global,
                    separate_global_qkv=separate_global_qkv,
                    global_dim_ratio=global_dim_ratio,
                    checkpoint_level=checkpoint_level,
                    use_relative_pos=use_relative_pos,
                    # initialization
                    attn_linear_init_mode=attn_linear_init_mode,
                    ffn_linear_init_mode=ffn_linear_init_mode,
                    norm_init_mode=norm_init_mode,
                ) for _ in range(depth[i])])
            self.cross_blocks.append(cross_block)

        # Construct upsampling layers
        if self.num_blocks > 1:
            if self.upsample_type == "upsample":
                self.upsample_layers = nn.ModuleList([
                    Upsample3DLayer(
                        dim=self.mem_shapes[i + 1][-1],
                        out_dim=self.mem_shapes[i][-1],
                        target_size=(target_temporal_length,) + self.mem_shapes[i][1:3],
                        kernel_size=upsample_kernel_size,
                        temporal_upsample=False,
                        conv_init_mode=conv_init_mode,
                    )
                    for i in range(self.num_blocks - 1)])
            else:
                raise NotImplementedError
            if self.hierarchical_pos_embed:
                self.hierarchical_pos_embed_l = nn.ModuleList([
                    PosEmbed(embed_dim=self.mem_shapes[i][-1], typ=pos_embed_type,
                             maxT=target_temporal_length, maxH=self.mem_shapes[i][1], maxW=self.mem_shapes[i][2])
                    for i in range(self.num_blocks - 1)])

        self.reset_parameters()

    def reset_parameters(self):
        for ms in self.self_blocks:
            for m in ms:
                m.reset_parameters()
        for ms in self.cross_blocks:
            for m in ms:
                m.reset_parameters()
        if self.num_blocks > 1:
            for m in self.upsample_layers:
                m.reset_parameters()
        if self.hierarchical_pos_embed:
            for m in self.hierarchical_pos_embed_l:
                m.reset_parameters()

    def forward(self, x, mem_l, mem_global_vector_l=None):
        """

        Parameters
        ----------
        x
            Shape (B, T_top, H_top, W_top, C)
        mem_l
            A list of memory tensors

        Returns
        -------
        out
        """
        B, T_top, H_top, W_top, C = x.shape
        assert T_top == self.target_temporal_length
        assert (H_top, W_top) == (self.mem_shapes[-1][1], self.mem_shapes[-1][2])
        for i in range(self.num_blocks - 1, -1, -1):
            mem_global_vector = None if mem_global_vector_l is None else mem_global_vector_l[i]
            if not self.use_first_self_attn and i == self.num_blocks - 1:
                # For the top block, we won't use the self attention layer and will directly use the cross attention layer.
                if i >= self.cross_start:
                    x = self.cross_blocks[i - self.cross_start][0](x, mem_l[i], mem_global_vector)
                for idx in range(self.depth[i] - 1):
                    if self.use_self_global:  # in this case `mem_global_vector` is guaranteed to be not None
                        if self.self_update_global:
                            x, mem_global_vector = self.self_blocks[i][idx](x, mem_global_vector)
                        else:
                            x, _ = self.self_blocks[i][idx](x, mem_global_vector)
                    else:
                        x = self.self_blocks[i][idx](x)
                    if i >= self.cross_start:
                        x = self.cross_blocks[i - self.cross_start][idx + 1](x, mem_l[i], mem_global_vector)
            else:
                for idx in range(self.depth[i]):
                    if self.use_self_global:
                        if self.self_update_global:
                            x, mem_global_vector = self.self_blocks[i][idx](x, mem_global_vector)
                        else:
                            x, _ = self.self_blocks[i][idx](x, mem_global_vector)
                    else:
                        x = self.self_blocks[i][idx](x)
                    if i >= self.cross_start:
                        x = self.cross_blocks[i - self.cross_start][idx](x, mem_l[i], mem_global_vector)
            # Upsample
            if i > 0:
                x = self.upsample_layers[i - 1](x)
                if self.hierarchical_pos_embed:
                    x = self.hierarchical_pos_embed_l[i - 1](x)
        return x

class InitialEncoder(nn.Module):
    def __init__(self,
                 dim,
                 out_dim,
                 downsample_scale: Union[int, Sequence[int]],
                 num_conv_layers=2,
                 activation='leaky',
                 padding_type='nearest',
                 conv_init_mode="0",
                 linear_init_mode="0",
                 norm_init_mode="0",
                 ):
        super(InitialEncoder, self).__init__()

        self.num_conv_layers = num_conv_layers
        self.conv_init_mode = conv_init_mode
        self.linear_init_mode = linear_init_mode
        self.norm_init_mode = norm_init_mode

        conv_block = []
        for i in range(num_conv_layers):
            if i == 0:
                conv_block.append(nn.Conv2d(kernel_size=(3, 3), padding=(1, 1),
                                            in_channels=dim, out_channels=out_dim))
                conv_block.append(nn.GroupNorm(16, out_dim))
                conv_block.append(get_activation(activation))
            else:
                conv_block.append(nn.Conv2d(kernel_size=(3, 3), padding=(1, 1),
                                            in_channels=out_dim, out_channels=out_dim))
                conv_block.append(nn.GroupNorm(16, out_dim))
                conv_block.append(get_activation(activation))

        self.conv_block = nn.Sequential(*conv_block)
        if isinstance(downsample_scale, int):
            patch_merge_downsample = (1, downsample_scale, downsample_scale)
        elif len(downsample_scale) == 2:
            patch_merge_downsample = (1, *downsample_scale)
        elif len(downsample_scale) == 3:
            patch_merge_downsample = tuple(downsample_scale)
        else:
            raise NotImplementedError(f"downsample_scale {downsample_scale} format not supported!")
        self.patch_merge = PatchMerging3D(
            dim=out_dim, out_dim=out_dim,
            padding_type=padding_type,
            downsample=patch_merge_downsample,
            linear_init_mode=linear_init_mode,
            norm_init_mode=norm_init_mode)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.children():
            apply_initialization(m,
                                 conv_mode=self.conv_init_mode,
                                 linear_mode=self.linear_init_mode,
                                 norm_mode=self.norm_init_mode)

    def forward(self, x):
        """

        x --> [K x Conv2D] --> PatchMerge

        Parameters
        ----------
        x
            Shape (B, T, H, W, C)

        Returns
        -------
        out
            Shape (B, T, H_new, W_new, C_out)
        """
        B, T, H, W, C = x.shape
        if self.num_conv_layers > 0:
            x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)
            x = self.conv_block(x).permute(0, 2, 3, 1)  # (B * T, H, W, C_new)
            x = self.patch_merge(x.reshape(B, T, H, W, -1))
        else:
            x = self.patch_merge(x)
        return x

class FinalDecoder(nn.Module):

    def __init__(self,
                 target_thw,
                 dim,
                 num_conv_layers=2,
                 activation='leaky',
                 conv_init_mode="0",
                 linear_init_mode="0",
                 norm_init_mode="0",
                 ):
        super(FinalDecoder, self).__init__()
        self.target_thw = target_thw
        self.dim = dim
        self.num_conv_layers = num_conv_layers
        self.conv_init_mode = conv_init_mode
        self.linear_init_mode = linear_init_mode
        self.norm_init_mode = norm_init_mode

        conv_block = []
        for i in range(num_conv_layers):
            conv_block.append(nn.Conv2d(kernel_size=(3, 3), padding=(1, 1), in_channels=dim, out_channels=dim))
            conv_block.append(nn.GroupNorm(16, dim))
            conv_block.append(get_activation(activation))
        self.conv_block = nn.Sequential(*conv_block)
        self.upsample = Upsample3DLayer(
            dim=dim, out_dim=dim,
            target_size=target_thw, kernel_size=3,
            conv_init_mode=conv_init_mode)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.children():
            apply_initialization(m,
                                 conv_mode=self.conv_init_mode,
                                 linear_mode=self.linear_init_mode,
                                 norm_mode=self.norm_init_mode)

    def forward(self, x):
        """

        x --> Upsample --> [K x Conv2D]

        Parameters
        ----------
        x
            Shape (B, T, H, W, C)

        Returns
        -------
        out
            Shape (B, T, H_new, W_new, C)
        """
        x = self.upsample(x)
        if self.num_conv_layers > 0:
            B, T, H, W, C = x.shape
            x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)
            x = self.conv_block(x).permute(0, 2, 3, 1).reshape(B, T, H, W, -1)
        return x

class InitialStackPatchMergingEncoder(nn.Module):

    def __init__(self,
                 num_merge: int,
                 in_dim,
                 out_dim_list,
                 downsample_scale_list,
                 num_conv_per_merge_list=None,
                 activation='leaky',
                 padding_type='nearest',
                 conv_init_mode="0",
                 linear_init_mode="0",
                 norm_init_mode="0",
                 ):
        super(InitialStackPatchMergingEncoder, self).__init__()

        self.conv_init_mode = conv_init_mode
        self.linear_init_mode = linear_init_mode
        self.norm_init_mode = norm_init_mode

        self.num_merge = num_merge
        self.in_dim = in_dim
        self.out_dim_list = out_dim_list[:num_merge]
        self.downsample_scale_list = downsample_scale_list[:num_merge]
        self.num_conv_per_merge_list = num_conv_per_merge_list
        self.num_group_list = [max(1, out_dim // 4) for out_dim in self.out_dim_list]

        self.conv_block_list = nn.ModuleList()
        self.patch_merge_list = nn.ModuleList()
        for i in range(num_merge):
            if i == 0:
                in_dim = in_dim
            else:
                in_dim = self.out_dim_list[i - 1]
            out_dim = self.out_dim_list[i]
            downsample_scale = self.downsample_scale_list[i]

            conv_block = []
            for j in range(self.num_conv_per_merge_list[i]):
                if j == 0:
                    conv_in_dim = in_dim
                else:
                    conv_in_dim = out_dim
                conv_block.append(nn.Conv2d(kernel_size=(3, 3), padding=(1, 1),
                                            in_channels=conv_in_dim, out_channels=out_dim))
                conv_block.append(nn.GroupNorm(self.num_group_list[i], out_dim))
                conv_block.append(get_activation(activation))

            conv_block = nn.Sequential(*conv_block)
            self.conv_block_list.append(conv_block)
            patch_merge = PatchMerging3D(
                dim=out_dim, out_dim=out_dim,
                padding_type=padding_type,
                downsample=(1, downsample_scale, downsample_scale),
                linear_init_mode=linear_init_mode,
                norm_init_mode=norm_init_mode)
            self.patch_merge_list.append(patch_merge)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.children():
            apply_initialization(m,
                                 conv_mode=self.conv_init_mode,
                                 linear_mode=self.linear_init_mode,
                                 norm_mode=self.norm_init_mode)

    def get_out_shape_list(self, input_shape):
        """
        T, H, W, C
        """
        out_shape_list = []
        for patch_merge in self.patch_merge_list:
            input_shape = patch_merge.get_out_shape(input_shape)
            out_shape_list.append(input_shape)
        return out_shape_list

    def forward(self, x):
        """

        x --> [K x Conv2D] --> PatchMerge --> ... --> [K x Conv2D] --> PatchMerge

        Parameters
        ----------
        x
            Shape (B, T, H, W, C)

        Returns
        -------
        out
            Shape (B, T, H_new, W_new, C_out)
        """
        for i, (conv_block, patch_merge) in \
                enumerate(zip(self.conv_block_list, self.patch_merge_list)):
            B, T, H, W, C = x.shape
            if self.num_conv_per_merge_list[i] > 0:
                x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)
                x = conv_block(x).permute(0, 2, 3, 1).reshape(B, T, H, W, -1)
            x = patch_merge(x)
        return x

class FinalStackUpsamplingDecoder(nn.Module):

    def __init__(self,
                 target_shape_list,
                 in_dim,
                 num_conv_per_up_list=None,
                 activation='leaky',
                 conv_init_mode="0",
                 linear_init_mode="0",
                 norm_init_mode="0",
                 ):
        """
        Parameters
        ----------
        target_shape_list:
            list of (T, H ,W ,C)
        """
        super(FinalStackUpsamplingDecoder, self).__init__()
        self.conv_init_mode = conv_init_mode
        self.linear_init_mode = linear_init_mode
        self.norm_init_mode = norm_init_mode

        self.target_shape_list = target_shape_list
        self.out_dim_list = [target_shape[-1] for target_shape in self.target_shape_list]
        self.num_upsample = len(target_shape_list)
        self.in_dim = in_dim
        self.num_conv_per_up_list = num_conv_per_up_list
        self.num_group_list = [max(1, out_dim // 4) for out_dim in self.out_dim_list]

        self.conv_block_list = nn.ModuleList()
        self.upsample_list = nn.ModuleList()
        for i in range(self.num_upsample):
            if i == 0:
                in_dim = in_dim
            else:
                in_dim = self.out_dim_list[i - 1]
            out_dim = self.out_dim_list[i]

            upsample = Upsample3DLayer(
                dim=in_dim, out_dim=in_dim,
                target_size=target_shape_list[i][:-1], kernel_size=3,
                conv_init_mode=conv_init_mode)
            self.upsample_list.append(upsample)
            conv_block = []
            for j in range(num_conv_per_up_list[i]):
                if j == 0:
                    conv_in_dim = in_dim
                else:
                    conv_in_dim = out_dim
                conv_block.append(nn.Conv2d(kernel_size=(3, 3), padding=(1, 1),
                                            in_channels=conv_in_dim, out_channels=out_dim))
                conv_block.append(nn.GroupNorm(self.num_group_list[i], out_dim))
                conv_block.append(get_activation(activation))
            conv_block = nn.Sequential(*conv_block)
            self.conv_block_list.append(conv_block)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.children():
            apply_initialization(m,
                                 conv_mode=self.conv_init_mode,
                                 linear_mode=self.linear_init_mode,
                                 norm_mode=self.norm_init_mode)

    @staticmethod
    def get_init_params(enc_input_shape, enc_out_shape_list, large_channel=False):
        dec_target_shape_list = list(enc_out_shape_list[:-1])[::-1] + [tuple(enc_input_shape), ]
        if large_channel:
            dec_target_shape_list_large_channel = []
            for i, enc_out_shape in enumerate(enc_out_shape_list[::-1]):
                dec_target_shape_large_channel = list(dec_target_shape_list[i])
                dec_target_shape_large_channel[-1] = enc_out_shape[-1]
                dec_target_shape_list_large_channel.append(tuple(dec_target_shape_large_channel))
            dec_target_shape_list = dec_target_shape_list_large_channel
        dec_in_dim = enc_out_shape_list[-1][-1]
        return dec_target_shape_list, dec_in_dim

    def forward(self, x):
        """

        x --> Upsample --> [K x Conv2D] --> ... --> Upsample --> [K x Conv2D]

        Parameters
        ----------
        x
            Shape (B, T, H, W, C)

        Returns
        -------
        out
            Shape (B, T, H_new, W_new, C)
        """
        for i, (conv_block, upsample) in \
                enumerate(zip(self.conv_block_list, self.upsample_list)):
            x = upsample(x)
            if self.num_conv_per_up_list[i] > 0:
                B, T, H, W, C = x.shape
                x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)
                x = conv_block(x).permute(0, 2, 3, 1).reshape(B, T, H, W, -1)
        return x

class CuboidTransformerModel(nn.Module):
    """Cuboid Transformer for spatiotemporal forecasting

    We adopt the Non-autoregressive encoder-decoder architecture.
    The decoder takes the multi-scale memory output from the encoder.

    The initial downsampling / upsampling layers will be
    Downsampling: [K x Conv2D --> PatchMerge]
    Upsampling: [Nearest Interpolation-based Upsample --> K x Conv2D]

    x --> downsample (optional) ---> (+pos_embed) ---> enc --> mem_l         initial_z (+pos_embed) ---> FC
                                                     |            |
                                                     |------------|
                                                           |
                                                           |
             y <--- upsample (optional) <--- dec <----------

    """
    def __init__(self,
                 input_shape,
                 target_shape,
                 base_units=128,
                 block_units=None,
                 scale_alpha=1.0,
                 num_heads=4,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 ffn_drop=0.0,
                 # inter-attn downsample/upsample
                 downsample=2,
                 downsample_type='patch_merge',
                 upsample_type="upsample",
                 upsample_kernel_size=3,
                 # encoder
                 enc_depth=[4, 4, 4],
                 enc_attn_patterns=None,
                 enc_cuboid_size=[(4, 4, 4), (4, 4, 4)],
                 enc_cuboid_strategy=[('l', 'l', 'l'), ('d', 'd', 'd')],
                 enc_shift_size=[(0, 0, 0), (0, 0, 0)],
                 enc_use_inter_ffn=True,
                 # decoder
                 dec_depth=[2, 2],
                 dec_cross_start=0,
                 dec_self_attn_patterns=None,
                 dec_self_cuboid_size=[(4, 4, 4), (4, 4, 4)],
                 dec_self_cuboid_strategy=[('l', 'l', 'l'), ('d', 'd', 'd')],
                 dec_self_shift_size=[(1, 1, 1), (0, 0, 0)],
                 dec_cross_attn_patterns=None,
                 dec_cross_cuboid_hw=[(4, 4), (4, 4)],
                 dec_cross_cuboid_strategy=[('l', 'l', 'l'), ('d', 'l', 'l')],
                 dec_cross_shift_hw=[(0, 0), (0, 0)],
                 dec_cross_n_temporal=[1, 2],
                 dec_cross_last_n_frames=None,
                 dec_use_inter_ffn=True,
                 dec_hierarchical_pos_embed=False,
                 # global vectors
                 num_global_vectors=4,
                 use_dec_self_global=True,
                 dec_self_update_global=True,
                 use_dec_cross_global=True,
                 use_global_vector_ffn=True,
                 use_global_self_attn=False,
                 separate_global_qkv=False,
                 global_dim_ratio=1,
                 z_init_method='nearest_interp',
                 # # initial downsample and final upsample
                 initial_downsample_type="conv",
                 initial_downsample_activation="leaky",
                 # initial_downsample_type=="conv"
                 initial_downsample_scale=1,
                 initial_downsample_conv_layers=2,
                 final_upsample_conv_layers=2,
                 # initial_downsample_type == "stack_conv"
                 initial_downsample_stack_conv_num_layers=1,
                 initial_downsample_stack_conv_dim_list=None,
                 initial_downsample_stack_conv_downscale_list=[1, ],
                 initial_downsample_stack_conv_num_conv_list=[2, ],
                 # # end of initial downsample and final upsample
                 ffn_activation='leaky',
                 gated_ffn=False,
                 norm_layer='layer_norm',
                 padding_type='ignore',
                 pos_embed_type='t+hw',
                 checkpoint_level=True,
                 use_relative_pos=True,
                 self_attn_use_final_proj=True,
                 dec_use_first_self_attn=False,
                 # initialization
                 attn_linear_init_mode="0",
                 ffn_linear_init_mode="0",
                 conv_init_mode="0",
                 down_up_linear_init_mode="0",
                 norm_init_mode="0",
                 ):
        """

        Parameters
        ----------
        input_shape
            Shape of the input tensor. It will be (T, H, W, C_in)
        target_shape
            Shape of the input tensor. It will be (T_out, H, W, C_out)
        base_units
            The base units
        z_init_method
            How the initial input to the decoder is initialized
        """
        super(CuboidTransformerModel, self).__init__()
        # initialization mode
        self.attn_linear_init_mode = attn_linear_init_mode
        self.ffn_linear_init_mode = ffn_linear_init_mode
        self.conv_init_mode = conv_init_mode
        self.down_up_linear_init_mode = down_up_linear_init_mode
        self.norm_init_mode = norm_init_mode

        assert len(enc_depth) == len(dec_depth)
        self.base_units = base_units
        self.num_global_vectors = num_global_vectors
        if global_dim_ratio != 1:
            assert separate_global_qkv == True, \
                f"Setting global_dim_ratio != 1 requires separate_global_qkv == True."
        self.global_dim_ratio = global_dim_ratio
        self.z_init_method = z_init_method
        assert self.z_init_method in ['zeros', 'nearest_interp', 'last', 'mean']

        self.input_shape = input_shape
        self.target_shape = target_shape
        T_in, H_in, W_in, C_in = input_shape
        T_out, H_out, W_out, C_out = target_shape
        assert H_in == H_out and W_in == W_out

        if self.num_global_vectors > 0:
            self.init_global_vectors = nn.Parameter(
                torch.zeros((self.num_global_vectors, global_dim_ratio*base_units)))

        new_input_shape = self.get_initial_encoder_final_decoder(
            initial_downsample_scale=initial_downsample_scale,
            initial_downsample_type=initial_downsample_type,
            activation=initial_downsample_activation,
            # initial_downsample_type=="conv"
            initial_downsample_conv_layers=initial_downsample_conv_layers,
            final_upsample_conv_layers=final_upsample_conv_layers,
            padding_type=padding_type,
            # initial_downsample_type == "stack_conv"
            initial_downsample_stack_conv_num_layers=initial_downsample_stack_conv_num_layers,
            initial_downsample_stack_conv_dim_list=initial_downsample_stack_conv_dim_list,
            initial_downsample_stack_conv_downscale_list=initial_downsample_stack_conv_downscale_list,
            initial_downsample_stack_conv_num_conv_list=initial_downsample_stack_conv_num_conv_list,
        )
        T_in, H_in, W_in, _ = new_input_shape

        self.encoder = CuboidTransformerEncoder(
            input_shape=(T_in, H_in, W_in, base_units),
            base_units=base_units,
            block_units=block_units,
            scale_alpha=scale_alpha,
            depth=enc_depth,
            downsample=downsample,
            downsample_type=downsample_type,
            block_attn_patterns=enc_attn_patterns,
            block_cuboid_size=enc_cuboid_size,
            block_strategy=enc_cuboid_strategy,
            block_shift_size=enc_shift_size,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            ffn_drop=ffn_drop,
            gated_ffn=gated_ffn,
            ffn_activation=ffn_activation,
            norm_layer=norm_layer,
            use_inter_ffn=enc_use_inter_ffn,
            padding_type=padding_type,
            use_global_vector=num_global_vectors > 0,
            use_global_vector_ffn=use_global_vector_ffn,
            use_global_self_attn=use_global_self_attn,
            separate_global_qkv=separate_global_qkv,
            global_dim_ratio=global_dim_ratio,
            checkpoint_level=checkpoint_level,
            use_relative_pos=use_relative_pos,
            self_attn_use_final_proj=self_attn_use_final_proj,
            # initialization
            attn_linear_init_mode=attn_linear_init_mode,
            ffn_linear_init_mode=ffn_linear_init_mode,
            conv_init_mode=conv_init_mode,
            down_linear_init_mode=down_up_linear_init_mode,
            norm_init_mode=norm_init_mode,
        )
        self.enc_pos_embed = PosEmbed(
            embed_dim=base_units, typ=pos_embed_type,
            maxH=H_in, maxW=W_in, maxT=T_in)
        mem_shapes = self.encoder.get_mem_shapes()

        self.z_proj = nn.Linear(mem_shapes[-1][-1], mem_shapes[-1][-1])
        self.dec_pos_embed = PosEmbed(
            embed_dim=mem_shapes[-1][-1], typ=pos_embed_type,
            maxT=T_out, maxH=mem_shapes[-1][1], maxW=mem_shapes[-1][2])
        self.decoder = CuboidTransformerDecoder(
            target_temporal_length=T_out,
            mem_shapes=mem_shapes,
            cross_start=dec_cross_start,
            depth=dec_depth,
            upsample_type=upsample_type,
            block_self_attn_patterns=dec_self_attn_patterns,
            block_self_cuboid_size=dec_self_cuboid_size,
            block_self_shift_size=dec_self_shift_size,
            block_self_cuboid_strategy=dec_self_cuboid_strategy,
            block_cross_attn_patterns=dec_cross_attn_patterns,
            block_cross_cuboid_hw=dec_cross_cuboid_hw,
            block_cross_shift_hw=dec_cross_shift_hw,
            block_cross_cuboid_strategy=dec_cross_cuboid_strategy,
            block_cross_n_temporal=dec_cross_n_temporal,
            cross_last_n_frames=dec_cross_last_n_frames,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            ffn_drop=ffn_drop,
            upsample_kernel_size=upsample_kernel_size,
            ffn_activation=ffn_activation,
            gated_ffn=gated_ffn,
            norm_layer=norm_layer,
            use_inter_ffn=dec_use_inter_ffn,
            max_temporal_relative=T_in + T_out,
            padding_type=padding_type,
            hierarchical_pos_embed=dec_hierarchical_pos_embed,
            pos_embed_type=pos_embed_type,
            use_self_global=(num_global_vectors > 0) and use_dec_self_global,
            self_update_global=dec_self_update_global,
            use_cross_global=(num_global_vectors > 0) and use_dec_cross_global,
            use_global_vector_ffn=use_global_vector_ffn,
            use_global_self_attn=use_global_self_attn,
            separate_global_qkv=separate_global_qkv,
            global_dim_ratio=global_dim_ratio,
            checkpoint_level=checkpoint_level,
            use_relative_pos=use_relative_pos,
            self_attn_use_final_proj=self_attn_use_final_proj,
            use_first_self_attn=dec_use_first_self_attn,
            # initialization
            attn_linear_init_mode=attn_linear_init_mode,
            ffn_linear_init_mode=ffn_linear_init_mode,
            conv_init_mode=conv_init_mode,
            up_linear_init_mode=down_up_linear_init_mode,
            norm_init_mode=norm_init_mode,
        )
        self.reset_parameters()

    def get_initial_encoder_final_decoder(
            self,
            initial_downsample_type,
            activation,
            # initial_downsample_type=="conv"
            initial_downsample_scale,
            initial_downsample_conv_layers,
            final_upsample_conv_layers,
            padding_type,
            # initial_downsample_type == "stack_conv"
            initial_downsample_stack_conv_num_layers,
            initial_downsample_stack_conv_dim_list,
            initial_downsample_stack_conv_downscale_list,
            initial_downsample_stack_conv_num_conv_list,
        ):
        T_in, H_in, W_in, C_in = self.input_shape
        T_out, H_out, W_out, C_out = self.target_shape
        # Construct the initial upsampling / downsampling layers
        self.initial_downsample_type = initial_downsample_type
        if self.initial_downsample_type == "conv":
            if isinstance(initial_downsample_scale, int):
                initial_downsample_scale = (1, initial_downsample_scale, initial_downsample_scale)
            elif len(initial_downsample_scale) == 2:
                initial_downsample_scale = (1, *initial_downsample_scale)
            elif len(initial_downsample_scale) == 3:
                initial_downsample_scale = tuple(initial_downsample_scale)
            else:
                raise NotImplementedError(f"initial_downsample_scale {initial_downsample_scale} format not supported!")
            # if any(ele > 1 for ele in initial_downsample_scale):
            self.initial_encoder = InitialEncoder(dim=C_in,
                                                  out_dim=self.base_units,
                                                  downsample_scale=initial_downsample_scale,
                                                  num_conv_layers=initial_downsample_conv_layers,
                                                  padding_type=padding_type,
                                                  activation=activation,
                                                  conv_init_mode=self.conv_init_mode,
                                                  linear_init_mode=self.down_up_linear_init_mode,
                                                  norm_init_mode=self.norm_init_mode)
            self.final_decoder = FinalDecoder(dim=self.base_units,
                                              target_thw=(T_out, H_out, W_out),
                                              num_conv_layers=final_upsample_conv_layers,
                                              activation=activation,
                                              conv_init_mode=self.conv_init_mode,
                                              linear_init_mode=self.down_up_linear_init_mode,
                                              norm_init_mode=self.norm_init_mode)
            new_input_shape = self.initial_encoder.patch_merge.get_out_shape(self.input_shape)
            self.dec_final_proj = nn.Linear(self.base_units, C_out)
            # else:
            #     self.initial_encoder = nn.Linear(C_in, self.base_units)
            #     self.final_decoder = nn.Identity()
            #     self.dec_final_proj = nn.Linear(self.base_units, C_out)
            #     new_input_shape = self.input_shape

        elif self.initial_downsample_type == "stack_conv":
            if initial_downsample_stack_conv_dim_list is None:
                initial_downsample_stack_conv_dim_list = [self.base_units, ] * initial_downsample_stack_conv_num_layers
            self.initial_encoder = InitialStackPatchMergingEncoder(
                num_merge=initial_downsample_stack_conv_num_layers,
                in_dim=C_in,
                out_dim_list=initial_downsample_stack_conv_dim_list,
                downsample_scale_list=initial_downsample_stack_conv_downscale_list,
                num_conv_per_merge_list=initial_downsample_stack_conv_num_conv_list,
                padding_type=padding_type,
                activation=activation,
                conv_init_mode=self.conv_init_mode,
                linear_init_mode=self.down_up_linear_init_mode,
                norm_init_mode=self.norm_init_mode)
            # use `self.target_shape` to get correct T_out
            initial_encoder_out_shape_list = self.initial_encoder.get_out_shape_list(self.target_shape)
            dec_target_shape_list, dec_in_dim = \
                FinalStackUpsamplingDecoder.get_init_params(
                    enc_input_shape=self.target_shape,
                    enc_out_shape_list=initial_encoder_out_shape_list,
                    large_channel=True)
            self.final_decoder = FinalStackUpsamplingDecoder(
                target_shape_list=dec_target_shape_list,
                in_dim=dec_in_dim,
                num_conv_per_up_list=initial_downsample_stack_conv_num_conv_list[::-1],
                activation=activation,
                conv_init_mode=self.conv_init_mode,
                linear_init_mode=self.down_up_linear_init_mode,
                norm_init_mode=self.norm_init_mode)
            self.dec_final_proj = nn.Linear(dec_target_shape_list[-1][-1], C_out)
            new_input_shape = self.initial_encoder.get_out_shape_list(self.input_shape)[-1]
        else:
            raise NotImplementedError
        self.input_shape_after_initial_downsample = new_input_shape
        T_in, H_in, W_in, _ = new_input_shape

        return new_input_shape

    def reset_parameters(self):
        if self.num_global_vectors > 0:
            nn.init.trunc_normal_(self.init_global_vectors, std=.02)
        if hasattr(self.initial_encoder, "reset_parameters"):
            self.initial_encoder.reset_parameters()
        else:
            apply_initialization(self.initial_encoder,
                                 conv_mode=self.conv_init_mode,
                                 linear_mode=self.down_up_linear_init_mode,
                                 norm_mode=self.norm_init_mode)
        if hasattr(self.final_decoder, "reset_parameters"):
            self.final_decoder.reset_parameters()
        else:
            apply_initialization(self.final_decoder,
                                 conv_mode=self.conv_init_mode,
                                 linear_mode=self.down_up_linear_init_mode,
                                 norm_mode=self.norm_init_mode)
        apply_initialization(self.dec_final_proj,
                             linear_mode=self.down_up_linear_init_mode)
        self.encoder.reset_parameters()
        self.enc_pos_embed.reset_parameters()
        self.decoder.reset_parameters()
        self.dec_pos_embed.reset_parameters()
        apply_initialization(self.z_proj,
                             linear_mode="0")

    def get_initial_z(self, final_mem, T_out):
        B = final_mem.shape[0]
        if self.z_init_method == 'zeros':
            z_shape = (1, T_out) + final_mem.shape[2:]
            initial_z = torch.zeros(z_shape, dtype=final_mem.dtype, device=final_mem.device)
            initial_z = self.z_proj(self.dec_pos_embed(initial_z)).expand(B, -1, -1, -1, -1)
        elif self.z_init_method == 'nearest_interp':
            # final_mem will have shape (B, T, H, W, C)
            initial_z = F.interpolate(final_mem.permute(0, 4, 1, 2, 3),
                                      size=(T_out, final_mem.shape[2], final_mem.shape[3])).permute(0, 2, 3, 4, 1)
            initial_z = self.z_proj(initial_z)
        elif self.z_init_method == 'last':
            initial_z = torch.broadcast_to(final_mem[:, -1:, :, :, :], (B, T_out) + final_mem.shape[2:])
            initial_z = self.z_proj(initial_z)
        elif self.z_init_method == 'mean':
            initial_z = torch.broadcast_to(final_mem.mean(axis=1, keepdims=True),
                                           (B, T_out) + final_mem.shape[2:])
            initial_z = self.z_proj(initial_z)
        else:
            raise NotImplementedError
        return initial_z

    def forward(self, x, verbose=False):
        """

        Parameters
        ----------
        x
            Shape (B, T, H, W, C)
        verbos
            if True, print intermediate shapes
        Returns
        -------
        out
            The output Shape (B, T_out, H, W, C_out)
        """
        B, _, _, _, _ = x.shape
        T_out = self.target_shape[0]
        x = self.initial_encoder(x)
        x = self.enc_pos_embed(x)
        if self.num_global_vectors > 0:
            init_global_vectors = self.init_global_vectors\
                .expand(B, self.num_global_vectors, self.global_dim_ratio*self.base_units)
            mem_l, mem_global_vector_l = self.encoder(x, init_global_vectors)
        else:
            mem_l = self.encoder(x)
        if verbose:
            for i, mem in enumerate(mem_l):
                print(f"mem[{i}].shape = {mem.shape}")
        initial_z = self.get_initial_z(final_mem=mem_l[-1],
                                       T_out=T_out)
        if self.num_global_vectors > 0:
            dec_out = self.decoder(initial_z, mem_l, mem_global_vector_l)
        else:
            dec_out = self.decoder(initial_z, mem_l)
        dec_out = self.final_decoder(dec_out)
        out = self.dec_final_proj(dec_out)
        return out
