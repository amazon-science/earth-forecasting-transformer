"""Patterns for cuboid self-attention / cross attention"""

import functools
from ..utils.registry import Registry

CuboidSelfAttentionPatterns = Registry('CuboidSelfAttentionPattern')
CuboidCrossAttentionPatterns = Registry('CuboidCrossAttentionPatterns')

# basic patterns

def full_attention(input_shape):
    T, H, W, _ = input_shape
    cuboid_size = [(T, H, W)]
    strategy = [('l', 'l', 'l')]
    shift_size = [(0, 0, 0)]
    return cuboid_size, strategy, shift_size

def self_axial(input_shape):
    """Axial attention proposed in https://arxiv.org/abs/1912.12180

    Parameters
    ----------
    input_shape
        T, H, W

    Returns
    -------
    cuboid_size
    strategy
    shift_size
    """
    T, H, W, _ = input_shape
    cuboid_size = [(T, 1, 1), (1, H, 1), (1, 1, W)]
    strategy = [('l', 'l', 'l'), ('l', 'l', 'l'), ('l', 'l', 'l')]
    shift_size = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
    return cuboid_size, strategy, shift_size

def self_video_swin(input_shape, P=2, M=4):
    """Adopt the strategy in Video SwinTransformer https://arxiv.org/pdf/2106.13230.pdf"""
    T, H, W, _ = input_shape
    P = min(P, T)
    M = min(M, H, W)
    cuboid_size = [(P, M, M), (P, M, M)]
    strategy = [('l', 'l', 'l'), ('l', 'l', 'l')]
    shift_size = [(0, 0, 0), (P // 2, M // 2, M // 2)]

    return cuboid_size, strategy, shift_size

def self_divided_space_time(input_shape):
    T, H, W, _ = input_shape
    cuboid_size = [(T, 1, 1), (1, H, W)]
    strategy = [('l', 'l', 'l'), ('l', 'l', 'l')]
    shift_size = [(0, 0, 0), (0, 0, 0)]
    return cuboid_size, strategy, shift_size

# basic patterns
CuboidSelfAttentionPatterns.register('full', full_attention)
CuboidSelfAttentionPatterns.register('axial', self_axial)
CuboidSelfAttentionPatterns.register('video_swin', self_video_swin)
CuboidSelfAttentionPatterns.register('divided_st', self_divided_space_time)
# video_swin_PxM
for p in [1, 2, 4, 8, 10]:
    for m in [1, 2, 4, 8, 16, 32]:
        CuboidSelfAttentionPatterns.register(
            f'video_swin_{p}x{m}',
            functools.partial(self_video_swin,
                              P=p, M=m))

# our proposals
def self_spatial_lg_v1(input_shape, M=4):
    T, H, W, _ = input_shape

    if H <= M and W <= M:
        cuboid_size = [(T, 1, 1), (1, H, W)]
        strategy = [('l', 'l', 'l'), ('l', 'l', 'l')]
        shift_size = [(0, 0, 0), (0, 0, 0)]
    else:
        cuboid_size = [(T, 1, 1), (1, M, M), (1, M, M)]
        strategy = [('l', 'l', 'l'), ('l', 'l', 'l'), ('d', 'd', 'd')]
        shift_size = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
    return cuboid_size, strategy, shift_size


# Following are our proposed new patterns based on the CuboidSelfAttention design.
CuboidSelfAttentionPatterns.register('spatial_lg_v1', self_spatial_lg_v1)
# spatial_lg
for m in [1, 2, 4, 8, 16, 32]:
    CuboidSelfAttentionPatterns.register(
        f'spatial_lg_{m}',
        functools.partial(self_spatial_lg_v1,
                          M=m))

def self_axial_space_dilate_K(input_shape, K=2):
    T, H, W, _ = input_shape
    K = min(K, H, W)
    cuboid_size = [(T, 1, 1),
                   (1, H // K, 1), (1, H // K, 1),
                   (1, 1, W // K), (1, 1, W // K)]
    strategy = [('l', 'l', 'l'),
                ('d', 'd', 'd'), ('l', 'l', 'l'),
                ('d', 'd', 'd'), ('l', 'l', 'l'),]
    shift_size = [(0, 0, 0),
                  (0, 0, 0), (0, 0, 0),
                  (0, 0, 0), (0, 0, 0)]
    return cuboid_size, strategy, shift_size
for k in [2, 4, 8]:
    CuboidSelfAttentionPatterns.register(
        f'axial_space_dilate_{k}',
        functools.partial(self_axial_space_dilate_K,
                          K=k))


def cross_KxK(mem_shape, K):
    """

    Parameters
    ----------
    mem_shape
    K

    Returns
    -------
    cuboid_hw
    shift_hw
    strategy
    n_temporal
    """
    T_mem, H, W, _ = mem_shape
    K = min(K, H, W)
    cuboid_hw = [(K, K)]
    shift_hw = [(0, 0)]
    strategy = [('l', 'l', 'l')]
    n_temporal = [1]
    return cuboid_hw, shift_hw, strategy, n_temporal

def cross_KxK_lg(mem_shape, K):
    """

    Parameters
    ----------
    mem_shape
    K

    Returns
    -------
    cuboid_hw
    shift_hw
    strategy
    n_temporal
    """
    T_mem, H, W, _ = mem_shape
    K = min(K, H, W)
    cuboid_hw = [(K, K), (K, K)]
    shift_hw = [(0, 0), (0, 0)]
    strategy = [('l', 'l', 'l'), ('d', 'd', 'd')]
    n_temporal = [1, 1]
    return cuboid_hw, shift_hw, strategy, n_temporal

def cross_KxK_heter(mem_shape, K):
    """

    Parameters
    ----------
    mem_shape
    K

    Returns
    -------
    cuboid_hw
    shift_hw
    strategy
    n_temporal
    """
    T_mem, H, W, _ = mem_shape
    K = min(K, H, W)
    cuboid_hw = [(K, K), (K, K), (K, K)]
    shift_hw = [(0, 0), (0, 0), (K // 2, K // 2)]
    strategy = [('l', 'l', 'l'), ('d', 'd', 'd'), ('l', 'l', 'l')]
    n_temporal = [1, 1, 1]
    return cuboid_hw, shift_hw, strategy, n_temporal

# # Our proposed CuboidCrossAttention patterns.
for k in [1, 2, 4, 8]:
    CuboidCrossAttentionPatterns.register(f'cross_{k}x{k}', functools.partial(cross_KxK, K=k))
    CuboidCrossAttentionPatterns.register(f'cross_{k}x{k}_lg', functools.partial(cross_KxK_lg, K=k))
    CuboidCrossAttentionPatterns.register(f'cross_{k}x{k}_heter', functools.partial(cross_KxK_heter, K=k))
