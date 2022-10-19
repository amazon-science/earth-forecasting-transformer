import os
from typing import List
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from ...utils.layout import change_layout_np
from .sevir_cmap import get_cmap, VIL_COLORS, VIL_LEVELS


HMF_COLORS = np.array([
    [82, 82, 82],
    [252, 141, 89],
    [255, 255, 191],
    [145, 191, 219]
]) / 255

THRESHOLDS = (0, 16, 74, 133, 160, 181, 219, 255)

def plot_hit_miss_fa(ax, y_true, y_pred, thres):
    mask = np.zeros_like(y_true)
    mask[np.logical_and(y_true >= thres, y_pred >= thres)] = 4
    mask[np.logical_and(y_true >= thres, y_pred < thres)] = 3
    mask[np.logical_and(y_true < thres, y_pred >= thres)] = 2
    mask[np.logical_and(y_true < thres, y_pred < thres)] = 1
    cmap = ListedColormap(HMF_COLORS)
    ax.imshow(mask, cmap=cmap)

def plot_hit_miss_fa_all_thresholds(ax, y_true, y_pred, **unused_kwargs):
    fig = np.zeros(y_true.shape)
    y_true_idx = np.searchsorted(THRESHOLDS, y_true)
    y_pred_idx = np.searchsorted(THRESHOLDS, y_pred)
    fig[y_true_idx == y_pred_idx] = 4
    fig[y_true_idx > y_pred_idx] = 3
    fig[y_true_idx < y_pred_idx] = 2
    # do not count results in these not challenging areas.
    fig[np.logical_and(y_true < THRESHOLDS[1], y_pred < THRESHOLDS[1])] = 1
    cmap = ListedColormap(HMF_COLORS)
    ax.imshow(fig, cmap=cmap)

def visualize_result(
        in_seq: np.array, target_seq: np.array,
        pred_seq_list: List[np.array], label_list: List[str],
        interval_real_time: float = 10.0, idx=0, norm=None, plot_stride=2,
        figsize=(24, 8), fs=10,
        vis_thresh=THRESHOLDS[2], vis_hits_misses_fas=True):
    """
    Parameters
    ----------
    model_list: list of nn.Module
    layout_list: list of str
    in_seq:     np.array
    target_seq: np.array
    interval_real_time: float
        The minutes of each plot interval
    """
    if norm is None:
        norm = {'scale': 255,
                'shift': 0}
    cmap_dict = lambda s: {'cmap': get_cmap(s, encoded=True)[0],
                           'norm': get_cmap(s, encoded=True)[1],
                           'vmin': get_cmap(s, encoded=True)[2],
                           'vmax': get_cmap(s, encoded=True)[3]}
    in_len = in_seq.shape[-1]
    out_len = target_seq.shape[-1]
    max_len = max(in_len, out_len)
    ncols = (max_len - 1) // plot_stride + 1
    if vis_hits_misses_fas:
        fig, ax = plt.subplots(nrows=2 + 3 * len(pred_seq_list),
                               ncols=ncols,
                               figsize=figsize)
    else:
        fig, ax = plt.subplots(nrows=2 + len(pred_seq_list),
                               ncols=ncols,
                               figsize=figsize)

    ax[0][0].set_ylabel('Inputs', fontsize=fs)
    for i in range(0, max_len, plot_stride):
        if i < in_len:
            xt = in_seq[idx, :, :, i] * norm['scale'] + norm['shift']
            ax[0][i // plot_stride].imshow(xt, **cmap_dict('vil'))
        else:
            ax[0][i // plot_stride].axis('off')

    ax[1][0].set_ylabel('Target', fontsize=fs)
    for i in range(0, max_len, plot_stride):
        if i < out_len:
            xt = target_seq[idx, :, :, i] * norm['scale'] + norm['shift']
            ax[1][i // plot_stride].imshow(xt, **cmap_dict('vil'))
            # ax[1][i // plot_stride].set_title(f'{5*(i+plot_stride)} Minutes')
        else:
            ax[1][i // plot_stride].axis('off')

    target_seq = target_seq[idx:idx + 1] * norm['scale'] + norm['shift']
    y_preds = [pred_seq[idx:idx + 1] * norm['scale'] + norm['shift']
               for pred_seq in pred_seq_list]

    # Plot model predictions
    if vis_hits_misses_fas:
        for k in range(len(pred_seq_list)):
            for i in range(0, max_len, plot_stride):
                if i < out_len:
                    ax[2 + 3 * k][i // plot_stride].imshow(y_preds[k][0, :, :, i], **cmap_dict('vil'))
                    plot_hit_miss_fa(ax[2 + 1 + 3 * k][i // plot_stride], target_seq[0, :, :, i], y_preds[k][0, :, :, i],
                                     vis_thresh)
                    plot_hit_miss_fa_all_thresholds(ax[2 + 2 + 3 * k][i // plot_stride], target_seq[0, :, :, i],
                                                    y_preds[k][0, :, :, i])
                else:
                    ax[2 + 3 * k][i // plot_stride].axis('off')
                    ax[2 + 1 + 3 * k][i // plot_stride].axis('off')
                    ax[2 + 2 + 3 * k][i // plot_stride].axis('off')

            ax[2 + 3 * k][0].set_ylabel(label_list[k] + '\nPrediction', fontsize=fs)
            ax[2 + 1 + 3 * k][0].set_ylabel(label_list[k] + f'\nScores\nThresh={vis_thresh}', fontsize=fs)
            ax[2 + 2 + 3 * k][0].set_ylabel(label_list[k] + '\nScores\nAll Thresh', fontsize=fs)
    else:
        for k in range(len(pred_seq_list)):
            for i in range(0, max_len, plot_stride):
                if i < out_len:
                    ax[2 + k][i // plot_stride].imshow(y_preds[k][0, :, :, i], **cmap_dict('vil'))
                else:
                    ax[2 + k][i // plot_stride].axis('off')

            ax[2 + k][0].set_ylabel(label_list[k] + '\nPrediction', fontsize=fs)

    for i in range(0, max_len, plot_stride):
        if i < out_len:
            ax[-1][i // plot_stride].set_title(f'{int(interval_real_time * (i + plot_stride))} Minutes', y=-0.25)

    for j in range(len(ax)):
        for i in range(len(ax[j])):
            ax[j][i].xaxis.set_ticks([])
            ax[j][i].yaxis.set_ticks([])

    # Legend of thresholds
    num_thresh_legend = len(VIL_LEVELS) - 1
    legend_elements = [Patch(facecolor=VIL_COLORS[i],
                             label=f'{int(VIL_LEVELS[i - 1])}-{int(VIL_LEVELS[i])}')
                       for i in range(1, num_thresh_legend + 1)]
    ax[0][0].legend(handles=legend_elements, loc='center left',
                    bbox_to_anchor=(-1.2, -0.),
                    borderaxespad=0, frameon=False, fontsize='10')
    if vis_hits_misses_fas:
        # Legend of Hit, Miss and False Alarm
        legend_elements = [Patch(facecolor=HMF_COLORS[3], edgecolor='k', label='Hit'),
                           Patch(facecolor=HMF_COLORS[2], edgecolor='k', label='Miss'),
                           Patch(facecolor=HMF_COLORS[1], edgecolor='k', label='False Alarm')]
        # ax[-1][0].legend(handles=legend_elements, loc='lower right',
        #                  bbox_to_anchor=(6., -.6),
        #                  ncol=5, borderaxespad=0, frameon=False, fontsize='16')
        ax[3][0].legend(handles=legend_elements, loc='center left',
                        bbox_to_anchor=(-2.2, -0.),
                        borderaxespad=0, frameon=False, fontsize='16')

    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    return fig, ax

def save_example_vis_results(
        save_dir, save_prefix, in_seq, target_seq, pred_seq, label,
        layout='NHWT', interval_real_time: float = 10.0, idx=0,
        plot_stride=2, fs=10, norm=None):
    """
    Parameters
    ----------
    in_seq: np.array
        float value 0-1
    target_seq: np.array
        float value 0-1
    pred_seq:   np.array
        float value 0-1
    interval_real_time: float
        The minutes of each plot interval
    """
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
