from typing import Optional, Union, Sequence, Dict
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, Subset, DataLoader, random_split
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
from einops import rearrange
from ..augmentation import TransformsFixRotation
from ...config import cfg


default_data_dir = os.path.join(cfg.datasets_dir, "earthnet2021")

def change_layout(data: np.ndarray,
                  in_layout: str = "HWCT",
                  out_layout: str = "HWCT"):
    axes = [None, ] * len(in_layout)
    for i, axis in enumerate(in_layout):
        axes[out_layout.find(axis)] = i
    return np.transpose(data, axes=axes)

def einops_change_layout(data: Union[np.ndarray, torch.Tensor],
                         einops_in_layout: str = "H W C T",
                         einops_out_layout: str = "H W C T"):
    return rearrange(data, f"{einops_in_layout} -> {einops_out_layout}")

class _BaseEarthNet2021Dataset(Dataset):
    r"""
    An .npy file contains a dict with

    "highresdynamic":   np.ndarray
        shape = (128, 128, 7, T_highres)
        channels are [blue, green, red, nir, cloud, scene, mask]
    "highresstatic":    np.ndarray
        shape = (128, 128, 1)
        channel is [elevation]
    "mesodynamic":      np.ndarray
        shape = (80, 80, 5, T_meso)
        channels are [precipitation, pressure, temp mean, temp min, temp max]
    "mesostatic":       np.ndarray
        shape = (80, 80, 1)
        channel is [elevation]

    train:
        T_highres = 30, T_meso = 150
    iid/ood test:
        T_highres = 10, T_meso = 150 for context
        T_highres = 20 for target
    extreme test:
        T_highres = 20, T_meso = 300 for context
        T_highres = 40 for target
    seasonal test:
        T_highres = 70, T_meso = 1050 for context
        T_highres = 140 for target
    """
    default_layout = "HWCT"
    default_static_layout = "HWC"
    # flip requires the last two dims to be H,W
    layout_for_aug = "CTHW"
    static_layout_for_aug = "CHW"

    def __init__(self,
                 return_mode: str = "default",
                 data_aug_mode: str = None,
                 data_aug_cfg: Dict = None,
                 layout: str = default_layout,
                 static_layout: str = default_static_layout,
                 highresstatic_expand_t: bool = False,
                 mesostatic_expand_t: bool = False,
                 meso_crop: Union[str, Sequence[Sequence[int]]] = None,
                 fp16: bool = False, ):
        r"""

        Parameters
        ----------
        return_mode:    str
            "default":
                return {
                    "highresdynamic": highresdynamic,
                    "highresstatic": highresstatic,
                    "mesodynamic": mesodynamic,
                    "mesostatic": mesostatic,
                }
            "minimal":
                return highresdynamic[..., 4, :], i.e., only RGB and IR channels.
        data_aug_mode:  str
            If None, no data augmentation is performed
            If "0", apply `RandomHorizontalFlip(p=0.5)` and RandomVerticalFlip(p=0.5)
        layout: str
            The layout of returned dynamic data ndarray.
        static_layout:  str
            The layout of returned static data ndarray. Take no effect if expanding temporal dim.
        highresstatic_expand_t: bool
            If True, add a new temporal dim for highresstatic data, use the same layout as dynamic data.
        mesostatic_expand_t:    bool
            If True, add a new temporal dim for mesostatic data, use the same layout as dynamic data.
        meso_crop:  Union[str, Sequence[Sequence[int]]]
            If None, take no effect
            If "default", use `((39, 41), (39, 41))` to crop out overlapping section with highres
            Can also be specified arbitrarily in form `((H_s, H_e), (W_s, W_e))`.
        fp16:   bool
            Use np.float16 if True else np.float32
        """
        self.return_mode = return_mode
        self.data_aug_mode = data_aug_mode
        if data_aug_cfg is None:
            data_aug_cfg = {}
        self.data_aug_cfg = data_aug_cfg
        if self.data_aug_mode is None:
            pass
        elif self.data_aug_mode in ["0", ]:
            self.data_aug = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
            ])
        elif self.data_aug_mode in ["1", "2"]:
            self.data_aug = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                TransformsFixRotation([0, 90, 180, 270]),
            ])
        else:
            raise NotImplementedError
        if layout is None:
            layout = self.default_layout
        self.layout = layout
        if static_layout is None:
            static_layout = self.default_static_layout
        self.static_layout = static_layout
        self.highresstatic_expand_t = highresstatic_expand_t
        self.mesostatic_expand_t = mesostatic_expand_t
        if isinstance(meso_crop, str):
            assert meso_crop == "default", f"meso_crop mode {meso_crop} not supported."
            meso_crop = ((39, 41), (39, 41))
        self.meso_crop = meso_crop
        self.np_dtype = np.float16 if fp16 else np.float32

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    @property
    def einops_default_layout(self):
        if not hasattr(self, "_einops_default_layout"):
            self._einops_default_layout = " ".join(self.default_layout)
        return self._einops_default_layout

    @property
    def einops_default_static_layout(self):
        if not hasattr(self, "_einops_default_static_layout"):
            self._einops_default_static_layout = " ".join(self.default_static_layout)
        return self._einops_default_static_layout

    @property
    def einops_layout(self):
        if not hasattr(self, "_einops_layout"):
            self._einops_layout = " ".join(self.layout)
        return self._einops_layout

    @property
    def einops_static_layout(self):
        if not hasattr(self, "_einops_static_layout"):
            self._einops_static_layout = " ".join(self.static_layout)
        return self._einops_static_layout

    @property
    def einops_layout_for_aug(self):
        if not hasattr(self, "_einops_layout_for_aug"):
            self._einops_layout_for_aug = " ".join(self.layout_for_aug)
        return self._einops_layout_for_aug

    @property
    def einops_static_layout_for_aug(self):
        if not hasattr(self, "_einops_static_layout_for_aug"):
            self._einops_static_layout_for_aug = " ".join(self.static_layout_for_aug)
        return self._einops_static_layout_for_aug

    def process_raw_data_from_npz(
            self,
            highresdynamic, highresstatic,
            mesodynamic, mesostatic):
        r"""
        Process np.ndarray data loaded from saved .npz files

        Parameters
        ----------
        highresdynamic:   np.ndarray
            shape = (128, 128, 7, T_highres)
            channels are [blue, green, red, nir, cloud, scene, mask]
        highresstatic:    np.ndarray
            shape = (128, 128, 1)
            channel is [elevation]
        mesodynamic:      np.ndarray
            shape = (80, 80, 5, T_meso)
            channels are [precipitation, pressure, temp mean, temp min, temp max]
        mesostatic:       np.ndarray
            shape = (80, 80, 1)
            channel is [elevation]

            (T_highres, T_meso) =
                (30, 150) for train, iid, ood
                (60, 300) for extreme
                (210, 1050) for seasonal

        Returns
        -------
        ret:    dict
        """
        T_highres = highresdynamic.shape[-1]
        T_meso = mesodynamic.shape[-1]

        if self.return_mode in ["default", ]:
            if self.meso_crop is not None:
                mesodynamic = self.crop_meso_spatial(mesodynamic)
                mesostatic = self.crop_meso_spatial(mesostatic)

            highresdynamic = np.nan_to_num(highresdynamic, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
            highresdynamic = np.clip(highresdynamic, a_min=0.0, a_max=1.0)
            mesodynamic = np.nan_to_num(mesodynamic, copy=False, nan=0.0)
            highresstatic = np.nan_to_num(highresstatic, copy=False, nan=0.0)
            mesostatic = np.nan_to_num(mesostatic, copy=False, nan=0.0)

            highresdynamic = einops_change_layout(
                data=highresdynamic,
                einops_in_layout=self.einops_default_layout,
                einops_out_layout=self.einops_layout)
            mesodynamic = einops_change_layout(
                data=mesodynamic,
                einops_in_layout=self.einops_default_layout,
                einops_out_layout=self.einops_layout)
            if self.highresstatic_expand_t:
                highresstatic = np.repeat(highresstatic[..., np.newaxis],
                                          repeats=T_highres,
                                          axis=-1)
                highresstatic = einops_change_layout(
                    data=highresstatic,
                    einops_in_layout=self.einops_default_layout,
                    einops_out_layout=self.einops_layout)
            else:
                highresstatic = einops_change_layout(
                    data=highresstatic,
                    einops_in_layout=self.einops_default_static_layout,
                    einops_out_layout=self.einops_static_layout)
            if self.mesostatic_expand_t:
                mesostatic = np.repeat(mesostatic[..., np.newaxis],
                                       repeats=T_meso,
                                       axis=-1)
                mesostatic = einops_change_layout(
                    data=mesostatic,
                    einops_in_layout=self.einops_default_layout,
                    einops_out_layout=self.einops_layout)
            else:
                mesostatic = einops_change_layout(
                    data=mesostatic,
                    einops_in_layout=self.einops_default_static_layout,
                    einops_out_layout=self.einops_static_layout)
            if self.return_mode == "default":
                if self.data_aug_mode is not None:
                    # TODO: augment all components consistently
                    raise NotImplementedError
                return {
                    "highresdynamic": highresdynamic,
                    "highresstatic": highresstatic,
                    "mesodynamic": mesodynamic,
                    "mesostatic": mesostatic,
                }
            else:
                raise NotImplementedError
        elif self.return_mode in ["minimal", ]:
            # only RGB, infrared channels and mask
            highresdynamic = np.nan_to_num(highresdynamic, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
            highresdynamic = np.clip(highresdynamic, a_min=0.0, a_max=1.0)

            if self.data_aug_mode is not None:
                highresdynamic = einops_change_layout(
                    data=highresdynamic,
                    einops_in_layout=self.einops_default_layout,
                    einops_out_layout=self.einops_layout_for_aug)
                highresdynamic = self.data_aug(torch.from_numpy(highresdynamic))
                highresdynamic = einops_change_layout(
                    data=highresdynamic,
                    einops_in_layout=self.einops_layout_for_aug,
                    einops_out_layout=self.einops_layout).numpy()
            else:
                highresdynamic = einops_change_layout(
                    data=highresdynamic,
                    einops_in_layout=self.einops_default_layout,
                    einops_out_layout=self.einops_layout)
            return highresdynamic
        else:
            raise NotImplementedError(f"return_mode {self.return_mode} not supported!")

    def crop_meso_spatial(self, meso_data):
        r"""
        Crop the meso data along spatial dims, under default layout.
        """
        if self.meso_crop is None:
            return meso_data
        else:
            return meso_data[
                   self.meso_crop[0][0]:self.meso_crop[0][1],
                   self.meso_crop[1][0]:self.meso_crop[1][1],
                   ...]

class EarthNet2021TrainDataset(_BaseEarthNet2021Dataset):
    r"""
    An .npy file contains a dict with
    "highresdynamic":   np.ndarray
        shape = (128, 128, 7, 30)
        channels are [blue, green, red, nir, cloud, scene, mask]
    "highresstatic":    np.ndarray
        shape = (128, 128, 1)
        channel is [elevation]
    "mesodynamic":      np.ndarray
        shape = (80, 80, 5, 150)
        channels are [precipitation, pressure, temp mean, temp min, temp max]
    "mesostatic":       np.ndarray
        shape = (80, 80, 1)
        channel is [elevation]
    """
    default_train_dir = os.path.join(default_data_dir, "train")

    T_highres = 30
    T_meso = 150

    def __init__(self,
                 return_mode: str = "default",
                 data_aug_mode: str = None,
                 data_aug_cfg: Dict = None,
                 data_dir: Union[Path, str] = None,
                 layout: str = None,
                 static_layout: str = None,
                 highresstatic_expand_t: bool = False,
                 mesostatic_expand_t: bool = False,
                 meso_crop: Union[str, Sequence[Sequence[int]]] = None,
                 fp16: bool = False):
        r"""

        Parameters
        ----------
        return_mode:    str
            "default":
                return {
                    "highresdynamic": highresdynamic,
                    "highresstatic": highresstatic,
                    "mesodynamic": mesodynamic,
                    "mesostatic": mesostatic,
                }
            "minimal":
                return highresdynamic[..., 4, :], i.e., only RGB and IR channels.
        data_aug_mode:  str
            If None, no data augmentation is performed
            If "0", apply `RandomHorizontalFlip(p=0.5)` and RandomVerticalFlip(p=0.5)
        data_aug_cfg:   dict
            dict which contains cfgs for controlling data augmentation
        data_dir:   Union[Path, str]
            Save dir of training data.
        layout: str
            The layout of returned dynamic data ndarray.
        static_layout:  str
            The layout of returned static data ndarray. Take no effect if expanding temporal dim.
        highresstatic_expand_t: bool
            If True, add a new temporal dim for highresstatic data, use the same layout as dynamic data.
        mesostatic_expand_t:    bool
            If True, add a new temporal dim for mesostatic data, use the same layout as dynamic data.
        meso_crop:  Union[str, Sequence[Sequence[int]]]
            If None, take no effect
            If "default", use `((39, 41), (39, 41))` to crop out overlapping section with highres
            Can also be specified arbitrarily in form `((H_s, H_e), (W_s, W_e))`.
        fp16:   bool
            Use np.float16 if True else np.float32
        """
        super(EarthNet2021TrainDataset, self).__init__(
            return_mode=return_mode,
            data_aug_mode=data_aug_mode,
            data_aug_cfg=data_aug_cfg,
            layout=layout,
            static_layout=static_layout,
            highresstatic_expand_t=highresstatic_expand_t,
            mesostatic_expand_t=mesostatic_expand_t,
            meso_crop=meso_crop,
            fp16=fp16, )
        if data_dir is None:
            data_dir = self.default_train_dir
        self.data_dir = Path(data_dir)
        self.npz_path_list = sorted(list(self.data_dir.glob("**/*.npz")))

    def __len__(self) -> int:
        return len(self.npz_path_list)

    def __getitem__(self, idx: int) -> dict:
        data_npz = np.load(self.npz_path_list[idx])

        # keep only [blue, green, red, nir, mask] channels
        highresdynamic = data_npz["highresdynamic"].astype(self.np_dtype)[:, :, [0, 1, 2, 3, 6], :]
        highresstatic = data_npz["highresstatic"].astype(self.np_dtype)
        mesodynamic = data_npz["mesodynamic"].astype(self.np_dtype)
        mesostatic = data_npz["mesostatic"].astype(self.np_dtype)

        if self.data_aug_mode in ["2", ]:
            processed_0 = self.process_raw_data_from_npz(
                highresdynamic, highresstatic,
                mesodynamic, mesostatic)
            data_npz_1 = np.load(self.npz_path_list[np.random.randint(len(self))])
            # keep only [blue, green, red, nir, mask] channels
            highresdynamic_1 = data_npz_1["highresdynamic"].astype(self.np_dtype)[:, :, [0, 1, 2, 3, 6], :]
            highresstatic_1 = data_npz_1["highresstatic"].astype(self.np_dtype)
            mesodynamic_1 = data_npz_1["mesodynamic"].astype(self.np_dtype)
            mesostatic_1 = data_npz_1["mesostatic"].astype(self.np_dtype)
            processed_1 = self.process_raw_data_from_npz(
                highresdynamic_1, highresstatic_1,
                mesodynamic_1, mesostatic_1)
            alpha = self.data_aug_cfg.get("mixup_alpha", 0.2)
            lam = np.random.beta(a=alpha, b=alpha)
            if self.return_mode in ["default", ]:
                ret = {}
                for key, val in processed_0.items():
                    ret[key] = lam * val + (1 - lam) * processed_1[key]
            elif self.return_mode in ["minimal", ]:
                ret = lam * processed_0 + (1 - lam) * processed_1
            else:
                raise NotImplementedError
            return ret
        else:
            return self.process_raw_data_from_npz(
                highresdynamic, highresstatic,
                mesodynamic, mesostatic)

class EarthNet2021TestDataset(_BaseEarthNet2021Dataset):
    r"""
    An .npy file contains a dict with
    "highresdynamic":   np.ndarray
        shape = (128, 128, 5, T_highres_context) for context
        shape = (128, 128, 5, T_highres_target) for target
        channels are [blue, green, red, nir, cloud, scene, mask]
    "highresstatic":    np.ndarray
        shape = (128, 128, 1)
        channel is [elevation]
    "mesodynamic":      np.ndarray
        shape = (80, 80, 5, T_meso) for context
    "mesostatic":       np.ndarray
        shape = (80, 80, 1)
        channel is [elevation]
    """
    default_iid_test_data_dir = os.path.join(default_data_dir, "iid_test_split")
    default_ood_test_data_dir = os.path.join(default_data_dir, "ood_test_split")
    default_extreme_test_data_dir = os.path.join(default_data_dir, "extreme_test_split")
    default_seasonal_test_data_dir = os.path.join(default_data_dir, "seasonal_test_split")

    T_highres_context = {"iid": 10, "ood": 10, "extreme": 20, "seasonal": 70}
    T_highres_target = {"iid": 20, "ood": 20, "extreme": 40, "seasonal": 140}
    T_highres = {key: val_context + val_target
                 for (key, val_context), (_, val_target) in
                 zip(T_highres_context.items(), T_highres_target.items())}
    T_meso = {"iid": 150, "ood": 150, "extreme": 300, "seasonal": 1050}

    def __init__(self,
                 return_mode: str = "default",
                 subset_name: str = "iid",
                 data_dir: Union[Path, str] = None,
                 layout: str = None,
                 static_layout: str = None,
                 highresstatic_expand_t: bool = False,
                 mesostatic_expand_t: bool = False,
                 meso_crop: Union[str, Sequence[Sequence[int]]] = None,
                 fp16: bool = False):
        r"""

        Parameters
        ----------
        return_mode:    str
            "default":
                return {
                    "highresdynamic": highresdynamic,
                    "highresstatic": highresstatic,
                    "mesodynamic": mesodynamic,
                    "mesostatic": mesostatic,
                }
            "minimal":
                return highresdynamic[..., 4, :], i.e., only RGB and IR channels.
        subset_name:    str
            Name of subset to load from default dir. Must be in ("iid", "ood", "extreme", "seasonal")
        data_dir:   Union[Path, str]
            if `subset_name` is None, use user specified dir
        layout: str
            The layout of returned dynamic data ndarray.
        static_layout:  str
            The layout of returned static data ndarray. Take no effect if expanding temporal dim.
        highresstatic_expand_t: bool
            If True, add a new temporal dim for highresstatic data, use the same layout as dynamic data.
        mesostatic_expand_t:    bool
            If True, add a new temporal dim for mesostatic data, use the same layout as dynamic data.
        meso_crop:  Union[str, Sequence[Sequence[int]]]
            If None, take no effect
            If "default", use `((39, 41), (39, 41))` to crop out overlapping section with highres
            Can also be specified arbitrarily in form `((H_s, H_e), (W_s, W_e))`.
        fp16:   bool
            Use np.float16 if True else np.float32
        """
        super(EarthNet2021TestDataset, self).__init__(
            return_mode=return_mode,
            data_aug_mode=None,
            layout=layout,
            static_layout=static_layout,
            highresstatic_expand_t=highresstatic_expand_t,
            mesostatic_expand_t=mesostatic_expand_t,
            meso_crop=meso_crop,
            fp16=fp16, )
        if subset_name == "iid":
            data_dir = self.default_iid_test_data_dir if data_dir is None else data_dir
        elif subset_name == "ood":
            data_dir = self.default_ood_test_data_dir if data_dir is None else data_dir
        elif subset_name == "extreme":
            data_dir = self.default_extreme_test_data_dir if data_dir is None else data_dir
        elif subset_name == "seasonal":
            data_dir = self.default_seasonal_test_data_dir if data_dir is None else data_dir
        else:
            assert subset_name is None  # Use user specified arg data_dir
        self.subset_name = subset_name
        self.data_dir = Path(data_dir)
        self.context_data_dir = self.data_dir.joinpath("context")
        self.target_data_dir = self.data_dir.joinpath("target")
        self.context_npz_path_list = sorted(list(self.context_data_dir.glob("**/*.npz")))
        self.target_npz_path_list = sorted(list(self.target_data_dir.glob("**/*.npz")))

    def __len__(self) -> int:
        return len(self.context_npz_path_list)

    def __getitem__(self, idx: int) -> dict:
        context_data_npz = np.load(self.context_npz_path_list[idx])
        target_data_npz = np.load(self.target_npz_path_list[idx])

        context_highresdynamic = context_data_npz["highresdynamic"].astype(self.np_dtype)
        highresstatic = context_data_npz["highresstatic"].astype(self.np_dtype)
        mesodynamic = context_data_npz["mesodynamic"].astype(self.np_dtype)
        mesostatic = context_data_npz["mesostatic"].astype(self.np_dtype)
        target_highresdynamic = target_data_npz["highresdynamic"].astype(self.np_dtype)
        highresdynamic = np.concatenate([context_highresdynamic,
                                         target_highresdynamic],
                                        axis=-1)

        return self.process_raw_data_from_npz(highresdynamic, highresstatic,
                                              mesodynamic, mesostatic)

class EarthNet2021LightningDataModule(LightningDataModule):

    def __init__(self,
                 return_mode: str = "default",
                 data_aug_mode: str = None,
                 data_aug_cfg: Dict = None,
                 train_data_dir: Union[Path, str] = None,
                 test_subset_name: Union[str, Sequence[str]] = ("iid", "ood"),
                 test_data_dir: Union[Union[Path, str], Sequence[Union[Path, str]]] = None,
                 val_ratio: float = 0.1,
                 train_val_split_seed: int = None,
                 layout: str = None,
                 static_layout: str = None,
                 highresstatic_expand_t: bool = False,
                 mesostatic_expand_t: bool = False,
                 meso_crop: Union[str, Sequence[Sequence[int]]] = None,
                 fp16: bool = False,
                 # datamodule_only
                 batch_size=1,
                 num_workers=8, ):
        super(EarthNet2021LightningDataModule, self).__init__()
        self.return_mode = return_mode
        self.data_aug_mode = data_aug_mode
        self.data_aug_cfg = data_aug_cfg
        if train_data_dir is None:
            train_data_dir = EarthNet2021TrainDataset.default_train_dir
        self.train_data_dir = train_data_dir

        if test_subset_name is None:
            if not isinstance(test_data_dir, Sequence):
                self.test_data_dir_list = [test_data_dir, ]
            else:
                self.test_data_dir_list = list(test_data_dir)
            self.test_subset_name_list = [None, ] * len(self.test_data_dir_list)
        else:
            if isinstance(test_subset_name, str):
                self.test_subset_name_list = [test_subset_name, ]
            elif isinstance(test_subset_name, Sequence):
                self.test_subset_name_list = list(test_subset_name)
            else:
                raise ValueError(f"Invalid type of test_subset_name {type(test_subset_name)}")
            self.test_data_dir_list = []
            for test_subset_name in self.test_subset_name_list:
                if test_subset_name == "iid":
                    test_data_dir = EarthNet2021TestDataset.default_iid_test_data_dir
                elif test_subset_name == "ood":
                    test_data_dir = EarthNet2021TestDataset.default_ood_test_data_dir
                elif test_subset_name == "extreme":
                    test_data_dir = EarthNet2021TestDataset.default_extreme_test_data_dir
                elif test_subset_name == "seasonal":
                    test_data_dir = EarthNet2021TestDataset.default_seasonal_test_data_dir
                else:
                    raise ValueError(f"Invalid test_subset_name {test_subset_name}")
                self.test_data_dir_list.append(test_data_dir)

        self.val_ratio = val_ratio
        self.train_val_split_seed = train_val_split_seed

        self.layout = layout
        self.static_layout = static_layout
        self.highresstatic_expand_t = highresstatic_expand_t
        self.mesostatic_expand_t = mesostatic_expand_t
        self.meso_crop = meso_crop
        self.fp16 = fp16
        # datamodule_only
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        assert os.path.exists(self.train_data_dir), "EarthNet2021 training set not found!"
        for test_data_dir in self.test_data_dir_list:
            assert os.path.exists(test_data_dir), f"EarthNet2021 test set at {test_data_dir} not found!"

    def setup(self, stage = None):
        if stage in (None, "fit"):
            train_val_data = EarthNet2021TrainDataset(
                return_mode=self.return_mode,
                data_aug_mode=self.data_aug_mode,
                data_aug_cfg=self.data_aug_cfg,
                data_dir=self.train_data_dir,
                layout=self.layout,
                static_layout=self.static_layout,
                highresstatic_expand_t=self.highresstatic_expand_t,
                mesostatic_expand_t=self.mesostatic_expand_t,
                meso_crop=self.meso_crop,
                fp16=self.fp16)
            val_size = int(self.val_ratio * len(train_val_data))
            train_size = len(train_val_data) - val_size

            if self.train_val_split_seed is not None:
                rnd_generator_dict = dict(generator=torch.Generator().manual_seed(self.train_val_split_seed))
            else:
                rnd_generator_dict = {}
            self.earthnet_train, self.earthnet_val = random_split(
                train_val_data, [train_size, val_size],
                **rnd_generator_dict)

        if stage in (None, "test"):
            self.earthnet_test_list = [
                EarthNet2021TestDataset(
                    return_mode=self.return_mode,
                    subset_name=test_subset_name,
                    data_dir=test_data_dir,
                    layout=self.layout,
                    static_layout=self.static_layout,
                    highresstatic_expand_t=self.highresstatic_expand_t,
                    mesostatic_expand_t=self.mesostatic_expand_t,
                    meso_crop=self.meso_crop,
                    fp16=self.fp16)
                for test_subset_name, test_data_dir in
                zip(self.test_subset_name_list, self.test_data_dir_list)]

        if stage in (None, "predict"):
            self.earthnet_predict_list = [
                EarthNet2021TestDataset(
                    return_mode=self.return_mode,
                    subset_name=test_subset_name,
                    data_dir=test_data_dir,
                    layout=self.layout,
                    static_layout=self.static_layout,
                    highresstatic_expand_t=self.highresstatic_expand_t,
                    mesostatic_expand_t=self.mesostatic_expand_t,
                    meso_crop=self.meso_crop,
                    fp16=self.fp16)
                for test_subset_name, test_data_dir in
                zip(self.test_subset_name_list, self.test_data_dir_list)]

    @property
    def num_train_samples(self):
        return len(self.earthnet_train)

    @property
    def num_val_samples(self):
        return len(self.earthnet_val)

    @property
    def num_test_samples(self):
        if len(self.earthnet_test_list) == 1:
            return len(self.earthnet_test_list[0])
        else:
            return [len(earthnet_test) for earthnet_test in self.earthnet_test_list]

    @property
    def num_predict_samples(self):
        if len(self.earthnet_predict_list) == 1:
            return len(self.earthnet_predict_list[0])
        else:
            return [len(earthnet_predict) for earthnet_predict in self.earthnet_predict_list]

    def train_dataloader(self):
        return DataLoader(self.earthnet_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.earthnet_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        if len(self.earthnet_test_list) == 1:
            return DataLoader(self.earthnet_test_list[0], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        else:
            return [DataLoader(earthnet_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
                    for earthnet_test in self.earthnet_test_list]

    def predict_dataloader(self):
        if len(self.earthnet_predict_list) == 1:
            return DataLoader(self.earthnet_predict_list[0], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        else:
            return [
                DataLoader(earthnet_predict, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
                for earthnet_predict in self.earthnet_predict_list]

def get_EarthNet2021_dataloaders(
        dataloader_return_mode: str = "default",
        data_aug_mode: str = None,
        data_aug_cfg: Dict = None,
        train_data_dir: Union[Path, str] = None,
        test_subset_name: Union[str, Sequence[str]] = ("iid", "ood"),
        test_data_dir: Union[Union[Path, str], Sequence[Union[Path, str]]] = None,
        val_ratio: float = 0.1,
        train_val_split_seed: int = None,
        layout: str = None,
        static_layout: str = None,
        highresstatic_expand_t: bool = False,
        mesostatic_expand_t: bool = False,
        meso_crop: Union[str, Sequence[Sequence[int]]] = None,
        fp16: bool = False,
        batch_size=1,
        num_workers=8, ):

    if test_subset_name is None:
        if not isinstance(test_data_dir, Sequence):
            test_data_dir_list = [test_data_dir, ]
        else:
            test_data_dir_list = list(test_data_dir)
        test_subset_name_list = [None, ] * len(test_data_dir_list)
    else:
        if isinstance(test_subset_name, str):
            test_subset_name_list = [test_subset_name, ]
        elif isinstance(test_subset_name, Sequence):
            test_subset_name_list = list(test_subset_name)
        else:
            raise ValueError(f"Invalid type of test_subset_name {type(test_subset_name)}")
        test_data_dir_list = []
        for test_subset_name in test_subset_name_list:
            if test_subset_name == "iid":
                test_data_dir = EarthNet2021TestDataset.default_iid_test_data_dir
            elif test_subset_name == "ood":
                test_data_dir = EarthNet2021TestDataset.default_ood_test_data_dir
            elif test_subset_name == "extreme":
                test_data_dir = EarthNet2021TestDataset.default_extreme_test_data_dir
            elif test_subset_name == "seasonal":
                test_data_dir = EarthNet2021TestDataset.default_seasonal_test_data_dir
            else:
                raise ValueError(f"Invalid test_subset_name {test_subset_name}")
            test_data_dir_list.append(test_data_dir)

    train_val_data = EarthNet2021TrainDataset(
        return_mode=dataloader_return_mode,
        data_aug_mode=data_aug_mode,
        data_aug_cfg=data_aug_cfg,
        data_dir=train_data_dir,
        layout=layout,
        static_layout=static_layout,
        highresstatic_expand_t=highresstatic_expand_t,
        mesostatic_expand_t=mesostatic_expand_t,
        meso_crop=meso_crop,
        fp16=fp16)
    val_size = int(val_ratio * len(train_val_data))
    train_size = len(train_val_data) - val_size

    if train_val_split_seed is not None:
        rnd_generator_dict = dict(generator=torch.Generator().manual_seed(train_val_split_seed))
    else:
        rnd_generator_dict = {}
    earthnet_train, earthnet_val = random_split(
        train_val_data, [train_size, val_size],
        **rnd_generator_dict)

    earthnet_test_list = [
        EarthNet2021TestDataset(
            return_mode=dataloader_return_mode,
            subset_name=test_subset_name,
            data_dir=test_data_dir,
            layout=layout,
            static_layout=static_layout,
            highresstatic_expand_t=highresstatic_expand_t,
            mesostatic_expand_t=mesostatic_expand_t,
            meso_crop=meso_crop,
            fp16=fp16)
        for test_subset_name, test_data_dir in
        zip(test_subset_name_list, test_data_dir_list)]

    num_test_samples = [len(earthnet_test) for earthnet_test in earthnet_test_list]
    test_dataloader = [DataLoader(earthnet_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                       for earthnet_test in earthnet_test_list]
    
    return {
        "train_dataloader": DataLoader(earthnet_train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val_dataloader": DataLoader(earthnet_val, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test_dataloader": test_dataloader,
        "num_train_samples": len(earthnet_train),
        "num_val_samples": len(earthnet_val),
        "num_test_samples": num_test_samples,
    }
