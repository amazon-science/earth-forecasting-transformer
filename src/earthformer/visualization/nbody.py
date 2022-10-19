import os
import numpy as np
from matplotlib import pyplot as plt
from ..utils.layout import change_layout_np


def save_example_vis_results(
        save_dir, save_prefix,
        in_seq, target_seq,
        pred_seq, label,
        layout='NHWT', idx=0,
        plot_stride=1, fs=10, norm="none"):
    r"""
    Parameters
    ----------
    in_seq: np.array
        float value 0-1
    target_seq: np.array
        float value 0-1
    pred_seq:   np.array or List[np.array]
        float value 0-1
    """
    in_seq = change_layout_np(in_seq, in_layout=layout).astype(np.float32)
    target_seq = change_layout_np(target_seq, in_layout=layout).astype(np.float32)
    if isinstance(pred_seq, list):
        pred_seq_list = [change_layout_np(ele, in_layout=layout).astype(np.float32)
                         for ele in pred_seq]
        assert isinstance(label, list) and len(label) == len(pred_seq)
    else:
        pred_seq_list = [change_layout_np(pred_seq, in_layout=layout).astype(np.float32), ]
        label_list = [label, ]
    fig_path = os.path.join(save_dir, f'{save_prefix}.png')
    if norm == "none":
        norm = {'scale': 1.0,
                'shift': 0.0}
    elif norm == "to255":
        norm = {'scale': 255,
                'shift': 0}
    else:
        raise NotImplementedError
    in_len = in_seq.shape[-1]
    out_len = target_seq.shape[-1]
    max_len = max(in_len, out_len)
    ncols = (max_len - 1) // plot_stride + 1
    fig, ax = plt.subplots(nrows=2 + len(pred_seq_list),
                           ncols=ncols,
                           figsize=(24, 8))

    ax[0][0].set_ylabel('Inputs\n', fontsize=fs)
    for i in range(0, max_len, plot_stride):
        if i < in_len:
            xt = in_seq[idx, :, :, i] * norm['scale'] + norm['shift']
            ax[0][i // plot_stride].imshow(xt, cmap='gray')
        else:
            ax[0][i // plot_stride].axis('off')

    ax[1][0].set_ylabel('Target\n', fontsize=fs)
    for i in range(0, max_len, plot_stride):
        if i < out_len:
            xt = target_seq[idx, :, :, i] * norm['scale'] + norm['shift']
            ax[1][i // plot_stride].imshow(xt, cmap='gray')
        else:
            ax[1][i // plot_stride].axis('off')

    y_preds = [pred_seq[idx:idx + 1] * norm['scale'] + norm['shift']
               for pred_seq in pred_seq_list]

    # Plot model predictions
    for k in range(len(pred_seq_list)):
        for i in range(0, max_len, plot_stride):
            if i < out_len:
                ax[2 + k][i // plot_stride].imshow(y_preds[k][0, :, :, i], cmap='gray')
            else:
                ax[2 + k][i // plot_stride].axis('off')

        ax[2 + k][0].set_ylabel(label_list[k], fontsize=fs)

    for i in range(0, max_len, plot_stride):
        if i < out_len:
            ax[-1][i // plot_stride].set_title(f"step {int(i + plot_stride)}", y=-0.25, fontsize=fs)

    for j in range(len(ax)):
        for i in range(len(ax[j])):
            ax[j][i].xaxis.set_ticks([])
            ax[j][i].yaxis.set_ticks([])

    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.savefig(fig_path)
    plt.close(fig)
