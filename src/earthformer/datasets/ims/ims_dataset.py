# TODO: maybe it is more efficient to read subsequent samples.
# TODO: allow to load sequences with more then 5 min apart
# TODO: verify that we took every important thing from SEVIR code

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
import pandas as pd
import numpy as np
import datetime, h5py, os
from typing import List, Union, Dict, Sequence
from src.earthformer.config import cfg

# IMS dataset constants
IMS_IMG_TYPES = {"MIDDLE_EAST_VIS", "MIDDLE_EAST_DAY_CLOUDS", "MIDDLE_EAST_COLORED", "MIDDLE_EAST_IR"}
IMS_RAW_DTYPES = {'MIDDLE_EAST_VIS': np.uint8}  # currently only VIS raw-type is known
IMS_DATA_SHAPE = {'MIDDLE_EAST_VIS': (600, 600, 4)}
PREPROCESS_SCALE_IMS = {'MIDDLE_EAST_VIS': 1 / 255}
PREPROCESS_OFFSET_IMS = {'MIDDLE_EAST_VIS': 0}
VALID_LAYOUTS = {'THWC'}

# IMS dataset directory
IMS_ROOT_DIR = os.path.join(cfg.datasets_dir, "ims")
IMS_CATALOG = os.path.join(IMS_ROOT_DIR, "CATALOG.csv")
IMS_DATA_DIR = os.path.join(IMS_ROOT_DIR, "data")


class IMSDataset(Dataset):
    def __init__(self,
                 img_type: str = 'MIDDLE_EAST_VIS',
                 seq_len: int = 49,
                 raw_seq_len: int = 169,
                 stride: int = 12,
                 layout: str = 'THWC',
                 ims_catalog: Union[str, pd.DataFrame] = None,
                 ims_data_dir: str = None,
                 start_date: datetime.datetime = None,
                 end_date: datetime.datetime = None,
                 shuffle: bool = False,
                 shuffle_seed: int = 1,
                 preprocess=None):

        super(IMSDataset, self).__init__()

        # files and directories parameters
        if ims_catalog is None:
            ims_catalog = IMS_CATALOG
        if ims_data_dir is None:
            ims_data_dir = IMS_DATA_DIR
        if isinstance(ims_catalog, str):
            self.catalog = pd.read_csv(ims_catalog, parse_dates=['time_utc'], low_memory=False)
        else:
            self.catalog = ims_catalog
        self.ims_data_dir = ims_data_dir

        # data parameters
        # TODO: consider including time filter.
        self.raw_seq_len = raw_seq_len
        assert img_type in IMS_IMG_TYPES, 'Invalid image type!'
        self.img_type = img_type
        self.start_date = start_date
        self.end_date = end_date
        if self.start_date is not None:
            self.catalog = self.catalog[self.catalog.time_utc > self.start_date]
        if self.end_date is not None:
            self.catalog = self.catalog[self.catalog.time_utc <= self.end_date]
        if layout not in VALID_LAYOUTS:
            raise ValueError(f'Invalid layout = {layout}! Must be one of {VALID_LAYOUTS}.')
        self.layout = layout
        if preprocess == None:
            preprocess = IMSPreprocess()

        # samples parameters
        assert seq_len <= self.raw_seq_len, f'seq_len must not be larger than raw_seq_len = {raw_seq_len}, got {seq_len}.'
        self.seq_len = seq_len
        self.stride = stride
        self.shuffle = shuffle
        self.shuffle_seed = int(shuffle_seed)
        self.preprocess = preprocess

        # setup
        self._events = None
        self._hdf_files = {}

        self._load_events()
        self._open_files()

    def _load_events(self):
        self._events = self.catalog[self.catalog.img_type == self.img_type]
        if self.shuffle:
            self._events = self._events.sample(frac=1, random_state=self.shuffle_seed)

    def _open_files(self):
        file_names = self._events['file_name'].unique()
        for f in file_names:
            self._hdf_files[f] = h5py.File(os.path.join(self.ims_data_dir, f), 'r')

    def _idx_sample(self, index):
        event_idx = index // self.num_seq_per_event
        seq_idx = index % self.num_seq_per_event
        event = self._events.iloc[event_idx]
        raw_seq = self._hdf_files[event['file_name']][self.img_type][event['file_index']]
        seq = raw_seq[slice(seq_idx * self.stride, seq_idx * self.stride + self.seq_len), :, :, :]  # TODO: allow layout different then THWC
        return seq

    def close(self):
        for f in self._hdf_files:
            self._hdf_files[f].close()
        self._hdf_files = {}

    @property
    def num_seq_per_event(self):
        return 1 + (self.raw_seq_len - self.seq_len) // self.stride

    @property
    def total_num_event(self):
        return int(self._events.shape[0])

    @property
    def total_num_seq(self):
        return int(self.num_seq_per_event * self.total_num_event)

    def __len__(self):
        return self.total_num_seq

    def __getitem__(self, index):
        sample = self._idx_sample(index)
        if self.preprocess:
            sample = self.preprocess(sample)
        return sample


class IMSPreprocess:
    # TODO: change the output data type
    def __init__(self, grayscale=False, crop={}, scale=True, data_type=torch.float32):
        # build the transformation function according to the parameters
        relevant_transforms = []

        # convert (H x W x C) to a Tensor (C x H x W),
        # either with scaling to [0.0, 1.0] or without
        relevant_transforms.append(transforms.ToTensor())
        if not scale:
            relevant_transforms.append(transforms.Lambda(lambda t: (t * 255).to(torch.uint8)))

        # convert to grayscale (1 x H x W) if necessary
        if grayscale: # TODO: when the image is akready in grayscale this creates bug
            relevant_transforms.append(transforms.Lambda(lambda x: x[:3, :, :]))
            relevant_transforms.append(transforms.Grayscale())

        # crop image if necessary
        if len(crop.keys()) > 0:
            relevant_transforms.append(transforms.Lambda(
                lambda t: F.crop(t, crop['top'], crop['left'], crop['height'], crop['width'])))

        # convert Tensor (C x H x W) to a Tensor (H x W x C)
        relevant_transforms.append(transforms.Lambda(
            lambda t: torch.moveaxis(t, -3, -1)))

        # convert data type
        relevant_transforms.append(transforms.Lambda(
            lambda t: t.to(data_type)))

        # save the final transformation function
        self.preprocess_frame = transforms.Compose(relevant_transforms)

    def preprocess_seq(self, seq):
        return torch.stack([self.preprocess_frame(frame) for frame in seq])

    def __call__(self, x):
        if x.ndim == 4:  # this is a seq
            return self.preprocess_seq(x)
        else:  # this is a frame - TODO: remove this later (now kept for debugging)
            return self.preprocess_frame(x)
