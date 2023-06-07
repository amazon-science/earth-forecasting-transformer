import os
from typing import List
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from ...utils.layout import change_layout_np
from .ims_cmap import get_cmap


def plot_seq(ax, row, label, seq, seq_len, max_len, idx, plot_stride, norm=None, fs=10):
    if norm is None:
        norm = {'scale': 255,
                'shift': 0}
        
    cmap_dict = lambda s: {'cmap': get_cmap(s, encoded=True)[0],
                           'norm': get_cmap(s, encoded=True)[1],
                           'vmin': get_cmap(s, encoded=True)[2],
                           'vmax': get_cmap(s, encoded=True)[3]}
    
    ax[row][0].set_ylabel(label, fontsize=fs)
    for i in range(0, max_len, plot_stride):
        if i < seq_len:
            xt = seq[idx, :, :, i] * norm['scale'] + norm['shift']
            ax[row][i // plot_stride].imshow(xt, **cmap_dict('vil'))
        else:
            ax[row][i // plot_stride].axis('off')

def visualize_result(
        in_seq: np.array, target_seq: np.array,
        pred_seq_list: List[np.array], label_list: List[str],
        interval_real_time: float = 10.0, idx=0, plot_stride=2,
        figsize=(24, 8),):
    
    in_len = in_seq.shape[-1]
    out_len = target_seq.shape[-1]
    max_len = max(in_len, out_len)
    nrows = (2 + len(pred_seq_list)) 
    ncols = (max_len - 1) // plot_stride + 1

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    plot_seq(ax, 0, "Inputs", 
             in_seq, in_len, max_len, idx, plot_stride)
    plot_seq(ax, 1, "Target", 
             target_seq, out_len, max_len, idx, plot_stride)
    for k in range(len(pred_seq_list)):
        plot_seq(ax, k+2, label_list[k] + '\nPrediction', 
                 pred_seq_list[k], out_len, max_len, idx, plot_stride)

    for i in range(0, max_len, plot_stride):
        if i < out_len:
            ax[-1][i // plot_stride].set_title(f'{int(interval_real_time * (i + plot_stride))} Minutes', y=-0.25)

    for j in range(len(ax)):
        for i in range(len(ax[j])):
            ax[j][i].xaxis.set_ticks([])
            ax[j][i].yaxis.set_ticks([])

    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    return fig, ax

def save_example_vis_results(
        save_dir, save_prefix, in_seq, target_seq, pred_seq, label,
        layout='NHWT', interval_real_time: float = 10.0, idx=0, plot_stride=2):

    in_seq = change_layout_np(in_seq, in_layout=layout).astype(np.float32)
    target_seq = change_layout_np(target_seq, in_layout=layout).astype(np.float32)
    pred_seq = change_layout_np(pred_seq, in_layout=layout).astype(np.float32)
    
    fig_path = os.path.join(save_dir, f'{save_prefix}.png')
    
    fig, ax = visualize_result(
        in_seq=in_seq, target_seq=target_seq, pred_seq_list=[pred_seq,],
        label_list=[label, ], interval_real_time=interval_real_time, idx=idx,
        plot_stride=plot_stride, fs=fs, norm=norm)
    
    plt.savefig(fig_path)
    plt.close(fig)
