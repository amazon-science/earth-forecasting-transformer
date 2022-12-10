"""CuboidTransformer adapted for auxiliary inputs in decoder"""
from typing import Sequence, Union
import warnings
from functools import lru_cache
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from .cuboid_transformer import (
    Upsample3DLayer, PatchMerging3D, PosEmbed,
    InitialEncoder, FinalDecoder,
    InitialStackPatchMergingEncoder, FinalStackUpsamplingDecoder,
    StackCuboidSelfAttentionBlock, StackCuboidCrossAttentionBlock,
    CuboidTransformerEncoder)
from .cuboid_transformer_patterns import CuboidSelfAttentionPatterns, CuboidCrossAttentionPatterns
from .utils import (
    get_activation, get_norm_layer,
    _generalize_padding, _generalize_unpadding,
    apply_initialization, round_to)


class CuboidTransformerUNetDecoder(nn.Module):
    """U-Net style Decoder of the CuboidTransformer.

    For each block, we first apply the StackCuboidSelfAttention and then apply the StackCuboidCrossAttention
    We add cross attention following 3 modes:

        cross_mode == "down":
            x --> attn --> cross_attn --> downscale --> ... --> z --> attn --> upscale --> ... --> out
                                ^                        ^
                                |                        |
                                |                        |
                               mem                      mem
        cross_mode == "up":
            x --> attn --> downscale --> ... --> z --> attn --> cross_attn --> upscale --> ... --> out
                                                                     ^                      ^
                                                                     |                      |
                                                                     |                      |
                                                                    mem                    mem
        cross_mode == "both":
            x --> attn --> cross_attn --> downscale --> ... --> z --> attn --> cross_attn --> upscale --> ... --> out
                                ^                        ^                          ^                      ^
                                |                        |                          |                      |
                                |                        |                          |                      |
                               mem                      mem                        mem                    mem
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
                 # different from `CuboidTransformerDecoder`, no arg `use_first_self_attn=False`
                 downsample=2,
                 downsample_type='patch_merge',
                 cross_mode="up",
                 down_linear_init_mode="0",
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
        downsample
            The downsample ratio
        downsample_type
            Type of the downsampling layer
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
        cross_last_n_frames
        cross_mode
            Must be one of ("up", "down", "both")
            Control whether the upsampling/downsampling/both phases cross attend to the encoded latent features
        num_heads
        attn_drop
        proj_drop
        ffn_drop
        ffn_activation
        gated_ffn
            Whether to enable gated ffn or not
        norm_layer
            The normalization layer
        use_inter_ffn
            Whether to use intermediate FFN
        hierarchical_pos_embed
            Whether to add pos embedding for each hierarchy.
        max_temporal_relative
        padding_type
        checkpoint_level
        """
        super(CuboidTransformerUNetDecoder, self).__init__()
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
        self.block_units = tuple(mem_shape[-1] for mem_shape in self.mem_shapes)
        self.depth = depth
        if not isinstance(downsample, (tuple, list)):
            downsample = (1, downsample, downsample)
        self.downsample = downsample
        self.downsample_type = downsample_type
        self.upsample_type = upsample_type
        self.hierarchical_pos_embed = hierarchical_pos_embed
        self.checkpoint_level = checkpoint_level
        self.use_self_global = use_self_global
        self.self_update_global = self_update_global
        self.use_cross_global = use_cross_global
        self.use_global_vector_ffn = use_global_vector_ffn

        assert cross_mode in ["up", "down", "both"], f"Invalid cross_mode {cross_mode}!"
        self.cross_mode = cross_mode
        self.up_use_cross = self.cross_mode in ["up", "both"]
        self.down_use_cross = self.cross_mode in ["down", "both"]

        if self.num_blocks > 1:
            # Construct downsampling layers
            if downsample_type == 'patch_merge':
                self.downsample_layers = nn.ModuleList(
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
        # Construct upsampling layers
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
                self.down_hierarchical_pos_embed_l = nn.ModuleList([
                    PosEmbed(embed_dim=self.block_units[i], typ=pos_embed_type,
                             maxT=self.mem_shapes[i][0], maxH=self.mem_shapes[i][1], maxW=self.mem_shapes[i][2])
                    for i in range(self.num_blocks - 1)])
                self.up_hierarchical_pos_embed_l = nn.ModuleList([
                    PosEmbed(embed_dim=self.block_units[i], typ=pos_embed_type,
                             maxT=self.mem_shapes[i][0], maxH=self.mem_shapes[i][1], maxW=self.mem_shapes[i][2])
                    for i in range(self.num_blocks - 1)])

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

        down_self_blocks = []
        up_self_blocks = []
        for i in range(self.num_blocks):
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
            down_self_blocks.append(nn.ModuleList(stack_cuboid_blocks))
            stack_cuboid_blocks = \
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
            up_self_blocks.append(nn.ModuleList(stack_cuboid_blocks))
        self.down_self_blocks = nn.ModuleList(down_self_blocks)
        self.up_self_blocks = nn.ModuleList(up_self_blocks)

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
        if self.up_use_cross:
            self.up_cross_blocks = nn.ModuleList()
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
                self.up_cross_blocks.append(cross_block)
        if self.down_use_cross:
            self.down_cross_blocks = nn.ModuleList()
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
                self.down_cross_blocks.append(cross_block)

        self.reset_parameters()

    def reset_parameters(self):
        for ms in self.down_self_blocks:
            for m in ms:
                m.reset_parameters()
        for ms in self.up_self_blocks:
            for m in ms:
                m.reset_parameters()
        if self.up_use_cross:
            for ms in self.up_cross_blocks:
                for m in ms:
                    m.reset_parameters()
        if self.down_use_cross:
            for ms in self.down_cross_blocks:
                for m in ms:
                    m.reset_parameters()
        if self.num_blocks > 1:
            for m in self.downsample_layers:
                m.reset_parameters()
            for m in self.upsample_layers:
                m.reset_parameters()
        if self.hierarchical_pos_embed:
            for m in self.down_hierarchical_pos_embed_l:
                m.reset_parameters()
            for m in self.up_hierarchical_pos_embed_l:
                m.reset_parameters()

    def forward(self, x, mem_l, mem_global_vector_l=None):
        """

        Parameters
        ----------
        x
            Shape (B, T, H, W, C)
        mem_l
            A list of memory tensors

        Returns
        -------
        out
        """
        B, T, H, W, C = x.shape
        assert T == self.target_temporal_length
        assert (H, W) == (self.mem_shapes[0][1], self.mem_shapes[0][2])
        new_mem_global_vector_l = []
        for i in range(self.num_blocks):
            # Downample
            if i > 0:
                x = self.downsample_layers[i - 1](x)
                if self.hierarchical_pos_embed:
                    x = self.down_hierarchical_pos_embed_l[i - 1](x)
            mem_global_vector = None if mem_global_vector_l is None else mem_global_vector_l[i]
            for idx in range(self.depth[i]):
                if self.use_self_global:
                    if self.self_update_global:
                        x, mem_global_vector = self.down_self_blocks[i][idx](x, mem_global_vector)
                    else:
                        x, _ = self.down_self_blocks[i][idx](x, mem_global_vector)
                else:
                    x = self.down_self_blocks[i][idx](x)
                if self.down_use_cross and i >= self.cross_start:
                    x = self.down_cross_blocks[i - self.cross_start][idx](x, mem_l[i], mem_global_vector)
                new_mem_global_vector_l.append(mem_global_vector)

        for i in range(self.num_blocks - 1, -1, -1):
            mem_global_vector = new_mem_global_vector_l[i]
            for idx in range(self.depth[i]):
                if self.use_self_global:
                    if self.self_update_global:
                        x, mem_global_vector = self.up_self_blocks[i][idx](x, mem_global_vector)
                    else:
                        x, _ = self.up_self_blocks[i][idx](x, mem_global_vector)
                else:
                    x = self.up_self_blocks[i][idx](x)
                if self.up_use_cross and i >= self.cross_start:
                    x = self.up_cross_blocks[i - self.cross_start][idx](x, mem_l[i], mem_global_vector)
            # Upsample
            if i > 0:
                x = self.upsample_layers[i - 1](x)
                if self.hierarchical_pos_embed:
                    x = self.up_hierarchical_pos_embed_l[i - 1](x)
        return x

class CuboidTransformerAuxModel(nn.Module):
    """Cuboid Transformer with auxiliary input in decoder for spatiotemporal forecasting

    We adopt the Non-autoregressive encoder-decoder architecture.
    The decoder takes the multi-scale memory output from the encoder, as well as auxiliary input.

    The initial downsampling / upsampling layers will be
    Downsampling: [K x Conv2D --> PatchMerge]
    Upsampling: [Nearest Interpolation-based Upsample --> K x Conv2D]

    x -----------> downsample (optional) ---> (+pos_embed) ---> enc ---------> mem_l
                                                             |                  |
                                                             |------------------|
                                                                          |
                                                                          |
    aux_input ---> downsample (optional) ---> (+pos_embed) ---> enc -> cross_attn -> dec -> upsample (optional) -> y
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
                 # initialization
                 attn_linear_init_mode="0",
                 ffn_linear_init_mode="0",
                 conv_init_mode="0",
                 down_up_linear_init_mode="0",
                 norm_init_mode="0",
                 # different from CuboidTransformerModel, no arg `dec_use_first_self_attn=False`
                 auxiliary_channels: int = 1,
                 unet_dec_cross_mode="up",
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
        """
        super(CuboidTransformerAuxModel, self).__init__()
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

        self.input_shape = input_shape
        self.target_shape = target_shape
        T_in, H_in, W_in, C_in = input_shape
        T_out, H_out, W_out, C_out = target_shape
        assert H_in == H_out and W_in == W_out
        self.auxiliary_channels = auxiliary_channels

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

        self.dec_pos_embed = PosEmbed(
            embed_dim=mem_shapes[-1][-1], typ=pos_embed_type,
            maxT=T_out, maxH=mem_shapes[-1][1], maxW=mem_shapes[-1][2])
        self.unet_dec_cross_mode = unet_dec_cross_mode
        self.decoder = CuboidTransformerUNetDecoder(
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
            # initialization
            attn_linear_init_mode=attn_linear_init_mode,
            ffn_linear_init_mode=ffn_linear_init_mode,
            conv_init_mode=conv_init_mode,
            up_linear_init_mode=down_up_linear_init_mode,
            norm_init_mode=norm_init_mode,
            # different from CuboidTransformerDecoder
            downsample=downsample,
            downsample_type=downsample_type,
            cross_mode=unet_dec_cross_mode,
            down_linear_init_mode=down_up_linear_init_mode,
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
            self.initial_aux_encoder = InitialEncoder(dim=self.auxiliary_channels,
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
            self.initial_aux_encoder = InitialStackPatchMergingEncoder(
                num_merge=initial_downsample_stack_conv_num_layers,
                in_dim=self.auxiliary_channels,
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

    def _interp_aux(self, aux_input, ref_shape):
        r"""
        Parameters
        ----------
        aux_input:  torch.Tensor
            Shape (B, T_aux, H_aux, W_aux, C_aux)
        ref_shape:  Sequence[int]
            (B, T, H, W, C)
        Returns
        -------
        ret:    torch.Tensor
            Shape (B, T, H, W, C_aux)
        """
        ret = rearrange(aux_input,
                        "b t h w c -> b c t h w")
        ret = F.interpolate(input=ret, size=ref_shape[:-1])
        ret = rearrange(ret,
                        "b c t h w -> b t h w c")
        return ret

    def interp_enc_aux(self, aux_input):
        return self._interp_aux(aux_input=aux_input, ref_shape=self.input_shape)

    def interp_dec_aux(self, aux_input):
        return self._interp_aux(aux_input=aux_input, ref_shape=self.target_shape)

    def forward(self, x, aux_enc, aux_dec, verbose=False):
        """

        Parameters
        ----------
        x:  torch.Tensor
            Shape (B, T, H, W, C)
        aux_enc:    torch.Tensor
            Shape (B, T, H_aux, W_aux, C_aux)
        aux_dec:    torch.Tensor
            Shape (B, T_out, H_aux, W_aux, C_aux)
        verbos
            if True, print intermediate shapes
        Returns
        -------
        out
            The output Shape (B, T_out, H, W, C_out)
        """
        B, _, _, _, _ = x.shape
        x = self.initial_encoder(x)
        aux_enc = self.initial_aux_encoder(self.interp_enc_aux(aux_enc))
        x = self.enc_pos_embed(x + aux_enc)
        if self.num_global_vectors > 0:
            init_global_vectors = self.init_global_vectors\
                .expand(B, self.num_global_vectors, self.global_dim_ratio*self.base_units)
            mem_l, mem_global_vector_l = self.encoder(x, init_global_vectors)
        else:
            mem_l = self.encoder(x)
        if verbose:
            for i, mem in enumerate(mem_l):
                print(f"mem[{i}].shape = {mem.shape}")
        aux_dec = self.initial_aux_encoder(self.interp_dec_aux(aux_dec))
        if self.num_global_vectors > 0:
            dec_out = self.decoder(aux_dec, mem_l, mem_global_vector_l)
        else:
            dec_out = self.decoder(aux_dec, mem_l)
        dec_out = self.final_decoder(dec_out)
        out = self.dec_final_proj(dec_out)
        return out
