"""Code is adapted from https://github.com/jerrywn121/TianChi_AIEarth/blob/main/SAConvLSTM/utils.py"""
import os
from typing import Optional
import numpy as np
from torch.utils.data import Dataset, DataLoader
import xarray as xr
from pathlib import Path
import torch
from pytorch_lightning import LightningDataModule
from ...config import cfg


NINO_WINDOW_T = 3  # Nino index is the sliding average over sst, window size is 3
CMIP6_SST_MAX = 10.198975563049316
CMIP6_SST_MIN = -16.549121856689453
CMIP5_SST_MAX = 8.991744995117188
CMIP5_SST_MIN = -9.33076286315918
CMIP6_NINO_MAX = 4.138188362121582
CMIP6_NINO_MIN = -3.5832221508026123
CMIP5_NINO_MAX = 3.8253555297851562
CMIP5_NINO_MIN = -2.691682815551758
SST_MAX = max(CMIP6_SST_MAX, CMIP5_SST_MAX)
SST_MIN = min(CMIP6_SST_MIN, CMIP5_SST_MIN)

default_enso_dir = os.path.join(cfg.datasets_dir, "icar_enso_2021")
# unzipped file is saved in `default_enso_nc_dir`.
default_enso_nc_dir = os.path.join(default_enso_dir, "enso_round1_train_20210201")

def scale_sst(sst):
    return (sst - SST_MIN) / (SST_MAX - SST_MIN)

def scale_back_sst(sst):
    return (SST_MAX - SST_MIN) * sst + SST_MIN

def prepare_inputs_targets(len_time, input_gap, input_length, pred_shift, pred_length, samples_gap):
    # input_gap=1: time gaps between two consecutive input frames
    # input_length=12: the number of input frames
    # pred_shift=26: the lead_time of the last target to be predicted
    # pred_length=26: the number of frames to be predicted
    assert pred_shift >= pred_length
    input_span = input_gap * (input_length - 1) + 1
    pred_gap = pred_shift // pred_length
    input_ind = np.arange(0, input_span, input_gap)
    target_ind = np.arange(0, pred_shift, pred_gap) + input_span + pred_gap - 1
    ind = np.concatenate([input_ind, target_ind]).reshape(1, input_length + pred_length)
    max_n_sample = len_time - (input_span + pred_shift - 1)
    ind = ind + np.arange(max_n_sample)[:, np.newaxis] @ np.ones((1, input_length + pred_length), dtype=int)
    return ind[::samples_gap]

def fold(data, size=36, stride=12):
    # inverse of unfold/sliding window operation
    # only applicable to the case where the size of the sliding windows is n*stride
    # data (N, size, *)
    # outdata (N_, *)
    # N/size is the number/width of sliding blocks
    assert size % stride == 0
    times = size // stride
    remain = (data.shape[0] - 1) % times
    if remain > 0:
        ls = list(data[::times]) + [data[-1, -(remain * stride):]]
        outdata = np.concatenate(ls, axis=0)  # (36*(151//3+1)+remain*stride, *, 15)
    else:
        outdata = np.concatenate(data[::times], axis=0)  # (36*(151/3+1), *, 15)
    assert outdata.shape[0] == size * ((data.shape[0] - 1) // times + 1) + remain * stride
    return outdata

def data_transform(data, num_years_per_model):
    # data (N, 36, *)
    # num_years_per_model: 151/140
    length = data.shape[0]
    assert length % num_years_per_model == 0
    num_models = length // num_years_per_model
    outdata = np.stack(np.split(data, length / num_years_per_model, axis=0), axis=-1)  # (151, 36, *, 15)
    # cmip6sst outdata.shape = (151, 36, 24, 48, 15) = (year, month, lat, lon, model)
    # cmip5sst outdata.shape = (140, 36, 24, 48, 17)
    # cmip6nino outdata.shape = (151, 36, 15)
    # cmip5nino outdata.shape = (140, 36, 17)
    outdata = fold(outdata, size=36, stride=12)
    # cmip6sst outdata.shape = (1836, 24, 48, 15), 1836 == 151 * 12 + 24
    # cmip5sst outdata.shape = (1704, 24, 48, 17)
    # cmip6nino outdata.shape = (1836, 15)
    # cmip5nino outdata.shape = (1704, 17)

    # check output data
    assert outdata.shape[-1] == num_models
    assert not np.any(np.isnan(outdata))
    return outdata

def read_raw_data(ds_dir, out_dir=None):
    # read and process raw cmip data from CMIP_train.nc and CMIP_label.nc
    train_cmip = xr.open_dataset(Path(ds_dir) / 'CMIP_train.nc').transpose('year', 'month', 'lat', 'lon')
    label_cmip = xr.open_dataset(Path(ds_dir) / 'CMIP_label.nc').transpose('year', 'month')
    # train_cmip.sst.values.shape = (4645, 36, 24, 48)

    # select longitudes
    lon = train_cmip.lon.values
    lon = lon[np.logical_and(lon >= 95, lon <= 330)]
    train_cmip = train_cmip.sel(lon=lon)

    cmip6sst = data_transform(data=train_cmip.sst.values[:2265],
                              num_years_per_model=151)
    cmip5sst = data_transform(data=train_cmip.sst.values[2265:],
                              num_years_per_model=140)
    cmip6nino = data_transform(data=label_cmip.nino.values[:2265],
                               num_years_per_model=151)
    cmip5nino = data_transform(data=label_cmip.nino.values[2265:],
                               num_years_per_model=140)
    # cmip6sst.shape = (1836, 24, 48, 15)
    # cmip5sst.shape = (1704, 24, 48, 17)
    assert len(cmip6sst.shape) == 4
    assert len(cmip5sst.shape) == 4
    assert len(cmip6nino.shape) == 2
    assert len(cmip5nino.shape) == 2
    # store processed data for faster data access
    if out_dir is not None:
        ds_cmip6 = xr.Dataset({'sst': (['month', 'lat', 'lon', 'model'], cmip6sst),
                               'nino': (['month', 'model'], cmip6nino)},
                              coords={'month': np.repeat(np.arange(1, 13)[None], cmip6nino.shape[0] // 12,
                                                         axis=0).flatten(),
                                      'lat': train_cmip.lat.values, 'lon': train_cmip.lon.values,
                                      'model': np.arange(15) + 1})
        ds_cmip6.to_netcdf(Path(out_dir) / 'cmip6.nc')
        ds_cmip5 = xr.Dataset({'sst': (['month', 'lat', 'lon', 'model'], cmip5sst),
                               'nino': (['month', 'model'], cmip5nino)},
                              coords={'month': np.repeat(np.arange(1, 13)[None], cmip5nino.shape[0] // 12,
                                                         axis=0).flatten(),
                                      'lat': train_cmip.lat.values, 'lon': train_cmip.lon.values,
                                      'model': np.arange(17) + 1})
        ds_cmip5.to_netcdf(Path(out_dir) / 'cmip5.nc')
    train_cmip.close()
    label_cmip.close()
    return cmip6sst, cmip5sst, cmip6nino, cmip5nino

def read_from_nc(ds_dir):
    # an alternative for reading processed data
    cmip6 = xr.open_dataset(Path(ds_dir) / 'cmip6.nc').transpose('month', 'lat', 'lon', 'model')
    cmip5 = xr.open_dataset(Path(ds_dir) / 'cmip5.nc').transpose('month', 'lat', 'lon', 'model')
    return cmip6.sst.values, cmip5.sst.values, cmip6.nino.values, cmip5.nino.values

def cat_over_last_dim(data):
    r"""
    treat different models (15 from CMIP6, 17 from CMIP5) as batch_size
    e.g., cmip6sst.shape = (178, 38, 24, 48, 15), converted_cmip6sst.shape = (2670, 38, 24, 48)
    e.g., cmip5sst.shape = (165, 38, 24, 48, 15), converted_cmip6sst.shape = (2475, 38, 24, 48)
    """
    return np.concatenate(np.moveaxis(data, -1, 0), axis=0)

class cmip_dataset(Dataset):

    def __init__(self,
                 sst_cmip6, nino_cmip6,
                 sst_cmip5, nino_cmip5,
                 samples_gap,
                 in_len=12,
                 out_len=26,
                 in_stride=1,
                 out_stride=1,
                 normalize_sst=True):
        r"""
        Parameters
        ----------
        sst_cmip6
        nino_cmip6
        sst_cmip5
        nino_cmip5
        samples_gap: int
            stride of seq sampling.
            e.g., samples_gap = 10, the first seq contains [0, 1, ..., T-1] frame indices, the second seq contains [10, 11, .., T+9]
        """
        super().__init__()
        self.normalize_sst = normalize_sst
        # cmip6 (N, *, 15)
        # cmip5 (N, *, 17)
        sst = []
        target_nino = []

        nino_idx_slice = slice(in_len, in_len + out_len - NINO_WINDOW_T + 1)  # e.g., 12:36
        if sst_cmip6 is not None:
            assert len(sst_cmip6.shape) == 4
            assert len(nino_cmip6.shape) == 2
            idx_sst = prepare_inputs_targets(
                len_time=sst_cmip6.shape[0],
                input_length=in_len, input_gap=in_stride,
                pred_shift=out_len * out_stride, pred_length=out_len,
                samples_gap=samples_gap)

            sst.append(cat_over_last_dim(sst_cmip6[idx_sst]))
            target_nino.append(cat_over_last_dim(nino_cmip6[idx_sst[:, nino_idx_slice]]))
        if sst_cmip5 is not None:
            assert len(sst_cmip5.shape) == 4
            assert len(nino_cmip5.shape) == 2
            idx_sst = prepare_inputs_targets(
                len_time=sst_cmip5.shape[0],
                input_length=in_len, input_gap=in_stride,
                pred_shift=out_len * out_stride, pred_length=out_len,
                samples_gap=samples_gap)
            sst.append(cat_over_last_dim(sst_cmip5[idx_sst]))
            target_nino.append(cat_over_last_dim(nino_cmip5[idx_sst[:, nino_idx_slice]]))

        # sst data containing both the input and target
        self.sst = np.concatenate(sst, axis=0)  # (N, in_len+out_len, lat, lon)
        if normalize_sst:
            self.sst = scale_sst(self.sst)
        # nino data containing the target only
        self.target_nino = np.concatenate(target_nino, axis=0)  # (N, out_len+NINO_WINDOW_T-1)
        assert self.sst.shape[0] == self.target_nino.shape[0]
        assert self.sst.shape[1] == in_len + out_len
        assert self.target_nino.shape[1] == out_len - NINO_WINDOW_T + 1

    def GetDataShape(self):
        return {'sst': self.sst.shape,
                'nino target': self.target_nino.shape}

    def __len__(self):
        return self.sst.shape[0]

    def __getitem__(self, idx):
        return self.sst[idx], self.target_nino[idx]

class ENSOLightningDataModule(LightningDataModule):

    def __init__(self,
                 data_dir=default_enso_nc_dir,
                 in_len=12,
                 out_len=26,
                 in_stride=1,
                 out_stride=1,
                 train_samples_gap=10,
                 eval_samples_gap=11,
                 normalize_sst=True,
                 # datamodule_only
                 batch_size=1,
                 num_workers=8, ):
        super(ENSOLightningDataModule, self).__init__()
        self.data_dir = data_dir

        self.in_len = in_len
        self.out_len = out_len
        self.in_stride = in_stride
        self.out_stride = out_stride
        self.train_samples_gap = train_samples_gap
        self.eval_samples_gap = eval_samples_gap
        self.normalize_sst = normalize_sst
        # datamodule_only
        self.batch_size = batch_size
        assert num_workers == 1, ValueError(f"Current implementation does not support `num_workers != 1`!")
        self.num_workers = num_workers

    def prepare_data(self):
        if not os.path.exists(self.data_dir):
            raise ValueError(
                f"ENSO data_dir {self.data_dir} not exits! Follow README.md to download and unzip dataset!")

    def setup(self, stage: Optional[str] = None):
        cmip6sst, cmip5sst, cmip6nino, cmip5nino = read_raw_data(self.data_dir)
        # TODO: more flexible train/val/test split
        sst_train = [cmip6sst, cmip5sst[..., :-2]]
        nino_train = [cmip6nino, cmip5nino[..., :-2]]
        sst_eval = [cmip5sst[..., -2:-1]]
        nino_eval = [cmip5nino[..., -2:-1]]
        sst_test = [cmip5sst[..., -1:]]
        nino_test = [cmip5nino[..., -1:]]

        if stage == "fit" or stage is None:
            self.enso_train = cmip_dataset(
                sst_cmip6=sst_train[0], nino_cmip6=nino_train[0],
                sst_cmip5=sst_train[1], nino_cmip5=nino_train[1],
                samples_gap=self.train_samples_gap,
                in_len=self.in_len,
                out_len=self.out_len,
                in_stride=self.in_stride,
                out_stride=self.out_stride,
                normalize_sst=self.normalize_sst)
            self.enso_val = cmip_dataset(
                sst_cmip6=None, nino_cmip6=None,
                sst_cmip5=sst_eval[0], nino_cmip5=nino_eval[0],
                samples_gap=self.eval_samples_gap,
                in_len=self.in_len,
                out_len=self.out_len,
                in_stride=self.in_stride,
                out_stride=self.out_stride,
                normalize_sst=self.normalize_sst)

        if stage == "test" or stage is None:
            self.enso_test = cmip_dataset(
                sst_cmip6=None, nino_cmip6=None,
                sst_cmip5=sst_test[0], nino_cmip5=nino_test[0],
                samples_gap=self.eval_samples_gap,
                in_len=self.in_len,
                out_len=self.out_len,
                in_stride=self.in_stride,
                out_stride=self.out_stride,
                normalize_sst=self.normalize_sst)

        if stage == "predict" or stage is None:
            self.enso_predict = cmip_dataset(
                sst_cmip6=None, nino_cmip6=None,
                sst_cmip5=sst_test[0], nino_cmip5=nino_test[0],
                samples_gap=self.eval_samples_gap,
                in_len=self.in_len,
                out_len=self.out_len,
                in_stride=self.in_stride,
                out_stride=self.out_stride,
                normalize_sst=self.normalize_sst)

    @property
    def num_train_samples(self):
        return len(self.enso_train)

    @property
    def num_val_samples(self):
        return len(self.enso_val)

    @property
    def num_test_samples(self):
        return len(self.enso_test)

    @property
    def num_predict_samples(self):
        return len(self.enso_predict)

    def train_dataloader(self):
        return DataLoader(self.enso_train,
                          shuffle=True,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.enso_val,
                          shuffle=False,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.enso_test,
                          shuffle=False,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.enso_predict,
                          shuffle=False,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)
