import os
from typing import List
import numpy as np
from matplotlib import pyplot as plt
from .ims_cmap import get_cmap

class IMSVisualize:
    def __init__(self, save_dir,
                 scale: bool = True,
                 fs: int = 10,
                 figsize: tuple = (24, 8),
                 plot_stride: int = 2,
                 ):
        self.save_dir = save_dir
        self.scale = scale
        self.fs = fs
        self.figsize = tuple(figsize)
        self.plot_stride = plot_stride

    def _plot_seq(self, ax, row, label, seq, seq_len, max_len):
        ax[row][0].set_ylabel(label, fontsize=self.fs)
        for i in range(0, max_len, self.plot_stride):
            if i < seq_len:
                xt = seq[i, :, :, :] * (255 if self.scale else 1)
                ax[row][i // self.plot_stride].imshow(xt)
            else:
                ax[row][i // self.plot_stride].axis('off')

    def _visualize_result(self,
                          in_seq, target_seq,
                          pred_seq_list,
                          label_list,
                          time_delta,
                          ):
        in_len = in_seq.shape[0]
        out_len = target_seq.shape[0]
        max_len = max(in_len, out_len)
        nrows = (2 + len(pred_seq_list))
        ncols = (max_len - 1) // self.plot_stride + 1

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=self.figsize)

        self._plot_seq(ax, 0, "Inputs", in_seq, in_len, max_len)
        self._plot_seq(ax, 1, "Target", target_seq, out_len, max_len)
        for k in range(len(pred_seq_list)):
            self._plot_seq(ax, k + 2, label_list[k] + "\nPrediction", pred_seq_list[k], out_len, max_len)

        for i in range(0, max_len, self.plot_stride):
            if i < max_len:
                ax[-1][i // self.plot_stride].set_title(f'{int(time_delta * (i + self.plot_stride))} Minutes', y=-0.25)

        # TODO: what is this???
        for i in range(nrows):
            for j in range(ncols):
                ax[i][j].xaxis.set_ticks([])
                ax[i][j].yaxis.set_ticks([])

        plt.subplots_adjust(hspace=0.05, wspace=0.05)
        return fig, ax

    def save_example(self,
                     save_prefix,
                     in_seq: np.array, target_seq: np.array, pred_seq_list: List[np.array],
                     label_list: List[str] = ["cuboid_ims"],
                     time_delta: float = 5.0,
                     ):
        # we assume the layout of our data is THWC
        fig_path = os.path.join(self.save_dir, f'{save_prefix}.png')

        fig, ax = self._visualize_result(
            in_seq=in_seq.astype(np.float32),
            target_seq=target_seq.astype(np.float32),
            pred_seq_list=[seq.astype(np.float32) for seq in pred_seq_list],
            label_list=label_list, time_delta=time_delta)

        plt.savefig(fig_path)
        plt.close(fig)
