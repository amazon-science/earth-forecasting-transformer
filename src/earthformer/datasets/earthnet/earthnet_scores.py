import numpy as np
from scipy.stats import hmean
import torch
from torchmetrics import Metric
from einops import rearrange
from .earthnet_toolkit.parallel_score import CubeCalculator as EN_CubeCalculator
from ...metrics.torchmetrics_wo_compute import MetricsUpdateWithoutCompute


class EarthNet2021Score(Metric):

    default_layout = "NHWCT"
    default_channel_axis = 3
    channels = 4

    def __init__(self,
                 layout: str = "NTHWC",
                 eps: float = 1e-4,
                 dist_sync_on_step: bool = False, ):
        super(EarthNet2021Score, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self.layout = layout
        self.eps = eps

        self.add_state("MAD",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("OLS",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("EMD",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("SSIM",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        # does not count if NaN
        self.add_state("num_MAD",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("num_OLS",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("num_EMD",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("num_SSIM",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")

    @property
    def einops_default_layout(self):
        if not hasattr(self, "_einops_default_layout"):
            self._einops_default_layout = " ".join(self.default_layout)
        return self._einops_default_layout

    @property
    def einops_layout(self):
        if not hasattr(self, "_einops_layout"):
            self._einops_layout = " ".join(self.layout)
        return self._einops_layout

    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None):
        r"""

        Parameters
        ----------
        pred, target:   torch.Tensor
            With the first dim as batch dim, and 4 channels (RGB and infrared)
        mask:   torch.Tensor
            With the first dim as batch dim, and 1 channel
        """
        pred_np = rearrange(pred.detach(), f"{self.einops_layout} -> {self.einops_default_layout}").cpu().numpy()
        target_np = rearrange(target.detach(), f"{self.einops_layout} -> {self.einops_default_layout}").cpu().numpy()
        # layout = "NHWCT"
        if mask is None:
            mask_np = np.ones_like(target_np)
        else:
            mask_np = torch.repeat_interleave(
                rearrange(1 - mask.detach(), f"{self.einops_layout} -> {self.einops_default_layout}"),
                repeats=self.channels, dim=self.default_channel_axis).cpu().numpy()
        for preds, targs, masks in zip(pred_np, target_np, mask_np):
            # Code is adapted from `load_file()` in ./earthnet_toolkit/parallel_score.py
            preds[preds < 0] = 0
            preds[preds > 1] = 1

            targs[np.isnan(targs)] = 0
            targs[targs > 1] = 1
            targs[targs < 0] = 0

            ndvi_preds = ((preds[:, :, 3, :] - preds[:, :, 2, :]) / (preds[:, :, 3, :] + preds[:, :, 2, :] + 1e-6))[:,
                         :, np.newaxis, :]
            ndvi_targs = ((targs[:, :, 3, :] - targs[:, :, 2, :]) / (targs[:, :, 3, :] + targs[:, :, 2, :] + 1e-6))[:,
                         :, np.newaxis, :]
            ndvi_masks = masks[:, :, 0, :][:, :, np.newaxis, :]
            # Code is adapted from `get_scores()` in ./earthnet_toolkit/parallel_score.py
            debug_info = {}
            mad, debug_info["MAD"] = EN_CubeCalculator.MAD(preds, targs, masks)
            ols, debug_info["OLS"] = EN_CubeCalculator.OLS(ndvi_preds, ndvi_targs, ndvi_masks)
            emd, debug_info["EMD"] = EN_CubeCalculator.EMD(ndvi_preds, ndvi_targs, ndvi_masks)
            ssim, debug_info["SSIM"] = EN_CubeCalculator.SSIM(preds, targs, masks)
            # does not count if NaN
            if mad is not None and not np.isnan(mad):
                self.MAD += mad
                self.num_MAD += 1
            if ols is not None and not np.isnan(ols):
                self.OLS += ols
                self.num_OLS += 1
            if emd is not None and not np.isnan(emd):
                self.EMD += emd
                self.num_EMD += 1
            if ssim is not None and not np.isnan(ssim):
                self.SSIM += ssim
                self.num_SSIM += 1

    def compute(self):
        MAD_mean = (self.MAD / (self.num_MAD + self.eps)).cpu().item()
        OLS_mean = (self.OLS / (self.num_OLS + self.eps)).cpu().item()
        EMD_mean = (self.EMD / (self.num_EMD + self.eps)).cpu().item()
        SSIM_mean = (self.SSIM / (self.num_SSIM + self.eps)).cpu().item()
        ENS = hmean([MAD_mean, OLS_mean, EMD_mean, SSIM_mean])
        return {
            "MAD": MAD_mean,
            "OLS": OLS_mean,
            "EMD": EMD_mean,
            "SSIM":SSIM_mean,
            "EarthNetScore": ENS,
        }

class EarthNet2021ScoreUpdateWithoutCompute(MetricsUpdateWithoutCompute):

    default_layout = "NHWCT"
    default_channel_axis = 3
    channels = 4

    def __init__(self,
                 layout: str = "NTHWC",
                 eps: float = 1e-4,
                 dist_sync_on_step: bool = False, ):
        super(EarthNet2021ScoreUpdateWithoutCompute, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self.layout = layout
        self.eps = eps

        self.add_state("MAD",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("OLS",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("EMD",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("SSIM",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        # does not count if NaN
        self.add_state("num_MAD",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("num_OLS",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("num_EMD",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("num_SSIM",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")

    @property
    def einops_default_layout(self):
        if not hasattr(self, "_einops_default_layout"):
            self._einops_default_layout = " ".join(self.default_layout)
        return self._einops_default_layout

    @property
    def einops_layout(self):
        if not hasattr(self, "_einops_layout"):
            self._einops_layout = " ".join(self.layout)
        return self._einops_layout

    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None):
        r"""

        Parameters
        ----------
        pred, target:   torch.Tensor
            With the first dim as batch dim, and 4 channels (RGB and infrared)
        mask:   torch.Tensor
            With the first dim as batch dim, and 1 channel
        """
        pred_np = rearrange(pred.detach(), f"{self.einops_layout} -> {self.einops_default_layout}").cpu().numpy()
        target_np = rearrange(target.detach(), f"{self.einops_layout} -> {self.einops_default_layout}").cpu().numpy()
        # layout = "NHWCT"
        if mask is None:
            mask_np = np.ones_like(target_np)
        else:
            mask_np = torch.repeat_interleave(
                rearrange(1 - mask.detach(), f"{self.einops_layout} -> {self.einops_default_layout}"),
                repeats=self.channels, dim=self.default_channel_axis).cpu().numpy()
        for preds, targs, masks in zip(pred_np, target_np, mask_np):
            # Code is adapted from `load_file()` in ./earthnet_toolkit/parallel_score.py
            preds[preds < 0] = 0
            preds[preds > 1] = 1

            targs[np.isnan(targs)] = 0
            targs[targs > 1] = 1
            targs[targs < 0] = 0

            ndvi_preds = ((preds[:, :, 3, :] - preds[:, :, 2, :]) / (preds[:, :, 3, :] + preds[:, :, 2, :] + 1e-6))[:,
                         :, np.newaxis, :]
            ndvi_targs = ((targs[:, :, 3, :] - targs[:, :, 2, :]) / (targs[:, :, 3, :] + targs[:, :, 2, :] + 1e-6))[:,
                         :, np.newaxis, :]
            ndvi_masks = masks[:, :, 0, :][:, :, np.newaxis, :]
            # Code is adapted from `get_scores()` in ./earthnet_toolkit/parallel_score.py
            debug_info = {}
            mad, debug_info["MAD"] = EN_CubeCalculator.MAD(preds, targs, masks)
            ols, debug_info["OLS"] = EN_CubeCalculator.OLS(ndvi_preds, ndvi_targs, ndvi_masks)
            emd, debug_info["EMD"] = EN_CubeCalculator.EMD(ndvi_preds, ndvi_targs, ndvi_masks)
            ssim, debug_info["SSIM"] = EN_CubeCalculator.SSIM(preds, targs, masks)
            # does not count if NaN
            if mad is not None and not np.isnan(mad):
                self.MAD += mad
                self.num_MAD += 1
            if ols is not None and not np.isnan(ols):
                self.OLS += ols
                self.num_OLS += 1
            if emd is not None and not np.isnan(emd):
                self.EMD += emd
                self.num_EMD += 1
            if ssim is not None and not np.isnan(ssim):
                self.SSIM += ssim
                self.num_SSIM += 1

    def compute(self):
        MAD_mean = (self.MAD / (self.num_MAD + self.eps)).cpu().item()
        OLS_mean = (self.OLS / (self.num_OLS + self.eps)).cpu().item()
        EMD_mean = (self.EMD / (self.num_EMD + self.eps)).cpu().item()
        SSIM_mean = (self.SSIM / (self.num_SSIM + self.eps)).cpu().item()
        ENS = hmean([MAD_mean, OLS_mean, EMD_mean, SSIM_mean])
        return {
            "MAD": MAD_mean,
            "OLS": OLS_mean,
            "EMD": EMD_mean,
            "SSIM":SSIM_mean,
            "EarthNetScore": ENS,
        }
