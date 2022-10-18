from typing import Tuple, Optional, Union
import numpy as np
import torch
from torchmetrics import Metric
from ..datasets.enso.enso_dataloader import NINO_WINDOW_T, scale_back_sst


def compute_enso_score(
        y_pred, y_true,
        acc_weight: Optional[Union[str, np.ndarray, torch.Tensor]] = None):
    r"""

    Parameters
    ----------
    y_pred: torch.Tensor
    y_true: torch.Tensor
    acc_weight: Optional[Union[str, np.ndarray, torch.Tensor]]
        None:   not used
        default:    use default acc_weight specified at https://tianchi.aliyun.com/competition/entrance/531871/information
        np.ndarray: custom weights

    Returns
    -------
    acc
    rmse
    """
    pred = y_pred - y_pred.mean(dim=0, keepdim=True)  # (N, 24)
    true = y_true - y_true.mean(dim=0, keepdim=True)  # (N, 24)
    cor = (pred * true).sum(dim=0) / (torch.sqrt(torch.sum(pred**2, dim=0) * torch.sum(true**2, dim=0)) + 1e-6)

    if acc_weight is None:
        acc = cor.sum()
    else:
        nino_out_len = y_true.shape[-1]
        if acc_weight == "default":
            acc_weight = torch.tensor([1.5] * 4 + [2] * 7 + [3] * 7 + [4] * (nino_out_len - 18))[:nino_out_len] \
                         * torch.log(torch.arange(nino_out_len) + 1)
        elif isinstance(acc_weight, np.ndarray):
            acc_weight = torch.from_numpy(acc_weight[:nino_out_len])
        elif isinstance(acc_weight, torch.Tensor):
            acc_weight = acc_weight[:nino_out_len]
        else:
            raise ValueError(f"Invalid acc_weight {acc_weight}!")
        acc_weight = acc_weight.to(y_pred)
        acc = (acc_weight * cor).sum()
    rmse = torch.mean((y_pred - y_true)**2, dim=0).sqrt().sum()
    return acc, rmse

def sst_to_nino(sst: torch.Tensor,
                normalize_sst: bool = True,
                detach: bool = True):
    r"""

    Parameters
    ----------
    sst:    torch.Tensor
        Shape = (N, T, H, W)

    Returns
    -------
    nino_index: torch.Tensor
        Shape = (N, T-NINO_WINDOW_T+1)
    """
    if detach:
        nino_index = sst.detach()
    else:
        nino_index = sst
    if normalize_sst:
        nino_index = scale_back_sst(nino_index)
    nino_index = nino_index[:, :, 10:13, 19:30].mean(dim=[2, 3])  # (N, 26)
    nino_index = nino_index.unfold(dimension=1, size=NINO_WINDOW_T, step=1).mean(dim=2)  # (N, 24)
    return nino_index

class ENSOScore(Metric):

    def __init__(self,
                 layout="NTHW",
                 out_len=26,
                 normalize_sst=True):
        super(ENSOScore, self).__init__()
        assert layout in ["NTHW", "NTHWC"], f"layout {layout} not supported"
        self.layout = layout
        self.normalize_sst = normalize_sst
        self.out_len = out_len
        self.nino_out_len = out_len - NINO_WINDOW_T + 1
        self.nino_weight = torch.from_numpy(np.array([1.5]*4 + [2]*7 + [3]*7 + [4]*(self.nino_out_len-18))
                                            * np.log(np.arange(self.nino_out_len)+1))

        self.add_state("sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_pixels", default=torch.tensor(0), dist_reduce_fx="sum")

        self.add_state("nino_preds", default=[], dist_reduce_fx="cat")
        self.add_state("nino_target", default=[], dist_reduce_fx="cat")
        # self.add_state("nino_preds", default=torch.zeros(0, 24), dist_reduce_fx="cat")
        # self.add_state("nino_target", default=torch.zeros(0, 24), dist_reduce_fx="cat")

    def update(self,
               preds: torch.Tensor, target: torch.Tensor,
               nino_preds:torch.Tensor = None, nino_target: torch.Tensor = None) -> None:
        r"""
        Parameters
        ----------
        preds
            Shape = (N, T, H, W) if self.layout == "NTHW"
                or  (N, T, H, W, 1) if self.layout == "NTHWC"
        target
            Shape = (N, T, H, W) if self.layout == "NTHW"
                or  (N, T, H, W, 1) if self.layout == "NTHWC"
        nino_preds
            Shape = (N, T-NINO_WINDOW_T+1)
        nino_target
            Shape = (N, T-NINO_WINDOW_T+1)
        Returns
        -------
        mse
        """
        if self.layout == "NTHWC":
            preds = preds[..., 0]
            target = target[..., 0]
        diff = preds - target
        sum_squared_error = torch.sum(diff * diff)
        num_pixels = target.numel()
        self.sum_squared_error += sum_squared_error
        self.num_pixels += num_pixels

        if nino_preds is None:
            nino_preds = sst_to_nino(sst=preds,
                                     normalize_sst=self.normalize_sst)
        if nino_target is None:
            nino_target = sst_to_nino(sst=target,
                                      normalize_sst=self.normalize_sst)
        nino_preds_list = [ele for ele in nino_preds]
        nino_target_list = [ele for ele in nino_target]
        self.nino_preds.extend(nino_preds_list)
        self.nino_target.extend(nino_target_list)
        # self.nino_preds = torch.cat((self.nino_preds, nino_preds))
        # self.nino_target = torch.cat((self.nino_target, nino_target))

    def compute(self) -> Tuple[float]:
        mse = self.sum_squared_error / self.num_pixels
        # print(f"self.nino_preds.shape = {self.nino_preds.shape}")
        y_pred=torch.stack(self.nino_preds, dim=0)
        y_true=torch.stack(self.nino_target, dim=0)
        # y_pred=self.nino_preds
        # y_true=self.nino_target
        acc, nino_rmse = compute_enso_score(
            y_pred=y_pred, y_true=y_true,
            acc_weight=self.nino_weight)
        return acc.cpu().item(), mse.cpu().item(),
