from typing import Optional
from math import ceil
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import matplotlib.colors as clr
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
import pandas as pd
from pathlib import Path
from .earthnet_toolkit.plot_cube import colorize, gallery
from .earthnet_dataloader import einops_change_layout


def vis_earthnet_seq(
        context_np=None,
        target_np=None,
        pred_np=None,
        batch_idx=0,
        ncols: int = 10,
        layout: str = "NHWCT",
        variable: str = "rgb",
        vegetation_mask=None,
        cloud_mask=True,
        save_path=None,
        dpi=300,
        figsize=None,
        font_size=10,
        y_label_rotation=0,
        y_label_offset=(-0.06, 0.4)):
    r"""
    Visualize the `batch_idx`-th seq in a batch of EarthNet data sequence

    Parameters
    ----------
    context_np: np.ndarray
        default_layout = "NHWCT"
    target_np:  np.ndarray
        default_layout = "NHWCT"
    pred_np:    np.ndarray
        default_layout = "NHWCT"
    ncols:  int
        Number of columns in plot
    layout: str
        The layout of np.ndarray
    variable:   str
        One of "rgb", "ndvi", "rr","pp","tg","tn","tx". Defaults to "rgb".
    vegetation_mask:   np.ndarray
        If given uses this as red mask over non-vegetation. S2GLC data. Defaults to None.
    cloud_mask: bool
        If True tries to use the last channel from the cubes sat imgs as blue cloud mask, 1 where no clouds, 0 where there are clouds. Defaults to True.
    save_path:  str
        If given, saves PNG to this path. Defaults to None.

    Returns
    -------
    fig:    plt.Figure
    """
    fontproperties = FontProperties()
    fontproperties.set_family('serif')
    # font.set_name('Times New Roman')
    fontproperties.set_size(font_size)
    # font.set_weight("bold")

    default_layout = "NTHWC"
    data_np_list = []
    label_list = []
    if context_np is not None:
        context_np = einops_change_layout(
            data=context_np,
            einops_in_layout=" ".join(layout),
            einops_out_layout=" ".join(default_layout))[batch_idx, ...]
        data_np_list.append(context_np)
        label_list.append("context")
    if target_np is not None:
        target_np = einops_change_layout(
            data=target_np,
            einops_in_layout=" ".join(layout),
            einops_out_layout=" ".join(default_layout))[batch_idx, ...]
        data_np_list.append(target_np)
        label_list.append("target")
    if pred_np is not None:
        pred_np = einops_change_layout(
            data=pred_np,
            einops_in_layout=" ".join(layout),
            einops_out_layout=" ".join(default_layout))[batch_idx, ...]
        data_np_list.append(pred_np)
        label_list.append("pred")

    fig, axes = plt.subplots(
        nrows=len(data_np_list),
        figsize=figsize, dpi=dpi,
        constrained_layout=True)
    for data, label, ax in zip(data_np_list, label_list, axes):
        if variable == "rgb":
            targ = np.stack([data[:, :, :, 2], data[:, :, :, 1], data[:, :, :, 0]], axis=-1)
            targ[targ < 0] = 0
            targ[targ > 0.5] = 0.5
            targ = 2 * targ
            if data.shape[-1] > 4 and cloud_mask:
                mask = data[:, :, :, -1]
                zeros = np.zeros_like(targ)
                zeros[:, :, :, 2] = 0.1
                targ = np.where(np.stack([mask] * 3, -1).astype(np.uint8) | np.isnan(targ).astype(np.uint8), zeros, targ)
            else:
                targ[np.isnan(targ)] = 0

        elif variable == "ndvi":
            if data.shape[-1] == 1:
                targ = data[:, :, :, 0]
            else:
                targ = (data[:, :, :, 3] - data[:, :, :, 2]) / (data[:, :, :, 2] + data[:, :, :, 3] + 1e-6)
            if data.shape[-1] > 4 and cloud_mask:
                cld_mask = 1 - data[:, :, :, -1]
            else:
                cld_mask = None

            if vegetation_mask is not None:
                if isinstance(vegetation_mask, str) or isinstance(vegetation_mask, Path):
                    vegetation_mask = np.load(vegetation_mask)
                if isinstance(vegetation_mask, np.lib.npyio.NpzFile):
                    vegetation_mask = vegetation_mask["landcover"]
                vegetation_mask = vegetation_mask.reshape(hw, hw)
                lc_mask = 1 - (vegetation_mask > 63) & (vegetation_mask < 105)
                lc_mask = np.repeat(lc_mask[np.newaxis, :, :], targ.shape[0], axis=0)
            else:
                lc_mask = None
            targ = colorize(targ, colormap="ndvi", mask_red=lc_mask, mask_blue=cld_mask)

        elif variable == "rr":
            targ = data[:, :, :, 0]
            targ = colorize(targ, colormap='Blues', mask_red=np.isnan(targ))
        elif variable == "pp":
            targ = data[:, :, :, 1]
            targ = colorize(targ, colormap='rainbow', mask_red=np.isnan(targ))
        elif variable in ["tg", "tn", "tx"]:
            targ = data[:, :, :, 2 if variable == "tg" else 3 if variable == "tn" else 4]
            targ = colorize(targ, colormap='coolwarm', mask_red=np.isnan(targ))
        else:
            raise ValueError(f"Invalid variable {variable}!")

        grid = gallery(targ, ncols=ncols)
        ax.set_ylabel(ylabel=label, fontproperties=fontproperties, rotation=y_label_rotation)
        ax.yaxis.set_label_coords(y_label_offset[0], y_label_offset[1])
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.imshow(grid)

    if variable != "rgb":
        colormap = \
        {"ndvi": "ndvi", "rr": "Blues", "pp": "rainbow", "tg": "coolwarm", "tn": "coolwarm", "tx": "coolwarm"}[variable]
        cmap = clr.LinearSegmentedColormap.from_list('ndvi',
                                                     ["#cbbe9a", "#fffde4", "#bccea5", "#66985b", "#2e6a32", "#123f1e",
                                                      "#0e371a", "#01140f", "#000d0a"],
                                                     N=256) if colormap == "ndvi" else copy.copy(plt.get_cmap(colormap))
        # divider = make_axes_locatable(plt.gca())
        # cax = divider.append_axes("right", size="5%", pad=0.1)
        vmin, vmax = \
        {"ndvi": (0, 1), "rr": (0, 50), "pp": (900, 1100), "tg": (-50, 50), "tn": (-50, 50), "tx": (-50, 50)}[variable]
        colorbar_label = {
            "ndvi": "NDVI", "rr": "Precipitation in mm/d", "pp": "Sea-level pressure in hPa",
            "tg": "Mean temperature in °C", "tn": "Minimum Temperature in °C", "tx": "Maximum Temperature in °C"
        }[variable]
        fig.colorbar(mappable=cm.ScalarMappable(norm=clr.Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
                     # cax=cax,
                     label=colorbar_label,
                     ax=axes, shrink=0.9, location="right")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parents[0].mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)

    return fig
