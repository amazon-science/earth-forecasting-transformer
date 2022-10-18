import warnings
from typing import Sequence
from shutil import copyfile
import inspect
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import Callback
from omegaconf import OmegaConf
import os
import argparse
from einops import rearrange
from pytorch_lightning import Trainer, seed_everything
from earthformer.config import cfg
from earthformer.utils.layout import layout_to_in_out_slice
from earthformer.baselines.persistence import Persistence
from earthformer.datasets.earthnet.earthnet_dataloader import EarthNet2021LightningDataModule, get_EarthNet2021_dataloaders
from earthformer.datasets.earthnet.earthnet_scores import EarthNet2021ScoreUpdateWithoutCompute
from earthformer.datasets.earthnet.visualization import vis_earthnet_seq
from earthformer.utils.apex_ddp import ApexDDPPlugin


class PersistenceEarthNet2021PLModule(pl.LightningModule):

    def __init__(self,
                 oc_file: str = None,
                 save_dir: str = None):
        super(PersistenceEarthNet2021PLModule, self).__init__()
        if oc_file is not None:
            oc_from_file = OmegaConf.load(open(oc_file, "r"))
        else:
            oc_from_file = None
        oc = self.get_base_config(oc_from_file=oc_from_file)
        self.save_hyperparameters(oc)
        self.oc = oc
        # layout
        self.in_len = oc.layout.in_len
        self.out_len = oc.layout.out_len
        self.layout = oc.layout.layout
        self.channel_axis = self.layout.find("C")
        self.batch_axis = self.layout.find("N")
        self.channels = oc.layout.data_channels
        # logging
        self.save_dir = save_dir
        self.logging_prefix = oc.logging.logging_prefix
        # visualization
        self.train_example_data_idx_list = list(oc.vis.train_example_data_idx_list)
        self.val_example_data_idx_list = list(oc.vis.val_example_data_idx_list)
        self.test_example_data_idx_list = list(oc.vis.test_example_data_idx_list)
        self.eval_example_only = oc.vis.eval_example_only

        self.torch_nn_module = Persistence(layout=self.layout)

        test_subset_name = oc.dataset.test_subset_name
        if isinstance(test_subset_name, Sequence):
            test_subset_name = list(test_subset_name)
        else:
            test_subset_name = [test_subset_name, ]
        self.test_subset_name = test_subset_name

        self.valid_mse = torchmetrics.MeanSquaredError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        self.valid_ens = EarthNet2021ScoreUpdateWithoutCompute(layout=self.layout, eps=1E-4)
        self.test_mse = torchmetrics.MeanSquaredError()
        self.test_mae = torchmetrics.MeanAbsoluteError()
        self.test_ens = EarthNet2021ScoreUpdateWithoutCompute(layout=self.layout, eps=1E-4)

        self.configure_save(cfg_file_path=oc_file)

    def configure_save(self, cfg_file_path=None):
        self.save_dir = os.path.join(cfg.exps_dir, self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self.scores_dir = os.path.join(self.save_dir, 'scores')
        os.makedirs(self.scores_dir, exist_ok=True)
        if cfg_file_path is not None:
            cfg_file_target_path = os.path.join(self.save_dir, "cfg.yaml")
            if (not os.path.exists(cfg_file_target_path)) or \
                    (not os.path.samefile(cfg_file_path, cfg_file_target_path)):
                copyfile(cfg_file_path, cfg_file_target_path)
        self.example_save_dir = os.path.join(self.save_dir, "examples")
        os.makedirs(self.example_save_dir, exist_ok=True)

    def get_base_config(self, oc_from_file=None):
        oc = OmegaConf.create()
        oc.layout = self.get_layout_config()
        oc.optim = self.get_optim_config()
        oc.logging = self.get_logging_config()
        oc.vis = self.get_vis_config()
        if oc_from_file is not None:
            # oc = apply_omegaconf_overrides(oc, oc_from_file)
            oc = OmegaConf.merge(oc, oc_from_file)
        return oc

    @staticmethod
    def get_layout_config():
        oc = OmegaConf.create()
        oc.in_len = 10
        oc.out_len = 20
        oc.layout = "NTHWC"
        oc.data_channels = 4
        return oc

    @classmethod
    def get_dataset_config(cls):
        cfg = OmegaConf.create()
        cfg.return_mode = "minimal"
        cfg.test_subset_name = ("iid", "ood")
        layout_cfg = cls.get_layout_config()
        cfg.in_len = layout_cfg.in_len
        cfg.out_len = layout_cfg.out_len
        cfg.layout = "THWC"
        cfg.static_layout = None
        cfg.val_ratio = 0.1
        cfg.train_val_split_seed = None
        cfg.highresstatic_expand_t = False
        cfg.mesostatic_expand_t = False
        cfg.meso_crop = None
        cfg.fp16 = False        
        return cfg

    @staticmethod
    def get_optim_config():
        cfg = OmegaConf.create()
        cfg.seed = None
        cfg.micro_batch_size = 8
        return cfg

    @staticmethod
    def get_logging_config():
        cfg = OmegaConf.create()
        cfg.logging_prefix = "EarthNet2021"
        cfg.use_wandb = False
        return cfg

    @staticmethod
    def get_vis_config():
        cfg = OmegaConf.create()
        cfg.train_example_data_idx_list = [0, ]
        cfg.val_example_data_idx_list = [0, ]
        cfg.test_example_data_idx_list = [0, ]
        cfg.eval_example_only = False
        cfg.ncols = 10
        cfg.dpi = 300
        cfg.figsize = None
        cfg.font_size = 10
        cfg.y_label_rotation = 0
        cfg.y_label_offset = (-0.05, 0.4)
        return cfg

    def set_trainer_kwargs(self, **kwargs):
        r"""
        Default kwargs used when initializing pl.Trainer
        """
        callbacks = kwargs.pop("callbacks", [])
        assert isinstance(callbacks, list)
        for ele in callbacks:
            assert isinstance(ele, Callback)

        logger = kwargs.pop("logger", [])
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=self.save_dir)
        csv_logger = pl_loggers.CSVLogger(save_dir=self.save_dir)
        logger += [tb_logger, csv_logger]
        if self.oc.logging.use_wandb:
            wandb_logger = pl_loggers.WandbLogger(project=self.oc.logging.logging_prefix,
                                                  save_dir=self.save_dir)
            logger += [wandb_logger, ]

        ret = dict(
            callbacks=callbacks,
            # log
            logger=logger,
            # save
            default_root_dir=self.save_dir,
            # ddp
            accelerator="gpu",
            # strategy="ddp",
            strategy=ApexDDPPlugin(find_unused_parameters=False, delay_allreduce=True),
            # optimization
            max_epochs=1,
            check_val_every_n_epoch=1,
            # NVIDIA amp
            precision=32,
        )
        return ret

    @staticmethod
    def get_earthnet2021_datamodule(
            dataset_cfg,
            micro_batch_size: int = 1,
            num_workers: int = 8):
        dm = EarthNet2021LightningDataModule(
            return_mode=dataset_cfg["return_mode"],
            val_ratio=dataset_cfg["val_ratio"],
            train_val_split_seed=dataset_cfg["train_val_split_seed"],
            layout=dataset_cfg["layout"],
            static_layout=dataset_cfg["static_layout"],
            highresstatic_expand_t=dataset_cfg["highresstatic_expand_t"],
            mesostatic_expand_t=dataset_cfg["mesostatic_expand_t"],
            meso_crop=dataset_cfg["meso_crop"],
            fp16=dataset_cfg["fp16"],
            # datamodule_only
            batch_size=micro_batch_size,
            num_workers=num_workers, )
        return dm

    @staticmethod
    def get_earthnet2021_dataloaders(
            dataset_cfg,
            micro_batch_size: int = 1,
            num_workers: int = 8):
        dataloader_dict = get_EarthNet2021_dataloaders(
            dataloader_return_mode=dataset_cfg["return_mode"],
            test_subset_name=dataset_cfg["test_subset_name"],
            val_ratio=dataset_cfg["val_ratio"],
            train_val_split_seed=dataset_cfg["train_val_split_seed"],
            layout=dataset_cfg["layout"],
            static_layout=dataset_cfg["static_layout"],
            highresstatic_expand_t=dataset_cfg["highresstatic_expand_t"],
            mesostatic_expand_t=dataset_cfg["mesostatic_expand_t"],
            meso_crop=dataset_cfg["meso_crop"],
            fp16=dataset_cfg["fp16"],
            batch_size=micro_batch_size,
            num_workers=num_workers, )
        return dataloader_dict

    @property
    def in_slice(self):
        if not hasattr(self, "_in_slice"):
            in_slice, out_slice = layout_to_in_out_slice(layout=self.layout,
                                                         in_len=self.in_len,
                                                         out_len=self.out_len)
            self._in_slice = in_slice
            self._out_slice = out_slice
        return self._in_slice

    @property
    def out_slice(self):
        if not hasattr(self, "_out_slice"):
            in_slice, out_slice = layout_to_in_out_slice(layout=self.layout,
                                                         in_len=self.in_len,
                                                         out_len=self.out_len)
            self._in_slice = in_slice
            self._out_slice = out_slice
        return self._out_slice

    def forward(self, batch):
        seq = batch[..., :self.channels]
        # mask = batch[..., self.channels:][self.out_slice].repeat_interleave(repeats=self.channels, axis=self.channel_axis)
        mask = batch[..., self.channels:][self.out_slice]
        # mask from dataloader: 1 for mask and 0 for non-masked
        in_seq = seq[self.in_slice]
        out_seq = seq[self.out_slice]
        output, _ = self.torch_nn_module(in_seq, out_seq)
        loss = F.mse_loss(output * (1 - mask), out_seq * (1 - mask))
        return output, loss

    def training_step(self, batch, batch_idx):
        seq = batch[..., :self.channels]
        mask = batch[..., self.channels:][self.out_slice]
        x = seq[self.in_slice]
        y = seq[self.out_slice]
        y_hat, loss = self(batch)
        micro_batch_size = x.shape[self.batch_axis]
        data_idx = int(batch_idx * micro_batch_size)
        if self.local_rank == 0:
            self.save_vis_step_end(
                data_idx=data_idx,
                context_seq=seq[self.in_slice].detach().float().cpu().numpy(),
                target_seq=seq[self.out_slice].detach().float().cpu().numpy(),
                pred_seq=y_hat.detach().float().cpu().numpy(),
                mode="train", )
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        seq = batch[..., :self.channels]
        mask = batch[..., self.channels:][self.out_slice]
        x = seq[self.in_slice]
        y = seq[self.out_slice]
        micro_batch_size = x.shape[self.batch_axis]
        data_idx = int(batch_idx * micro_batch_size)
        if not self.eval_example_only or data_idx in self.val_example_data_idx_list:
            y_hat, _ = self(batch)
            if self.local_rank == 0:
                self.save_vis_step_end(
                    data_idx=data_idx,
                    context_seq=seq[self.in_slice].detach().float().cpu().numpy(),
                    target_seq=seq[self.out_slice].detach().float().cpu().numpy(),
                    pred_seq=y_hat.detach().float().cpu().numpy(),
                    mode="val", )
            if self.precision == 16:
                y_hat = y_hat.float()
            self.valid_mse(y_hat * (1 - mask), y * (1 - mask))
            self.valid_mae(y_hat * (1 - mask), y * (1 - mask))
            self.valid_ens(y_hat, y, mask)
        return None

    def validation_epoch_end(self, outputs):
        valid_mse = self.valid_mse.compute()
        valid_mae = self.valid_mae.compute()
        valid_ens_dict = self.valid_ens.compute()
        valid_loss = -valid_ens_dict["EarthNetScore"]

        self.log('valid_loss_epoch', valid_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('valid_mse_epoch', valid_mse, prog_bar=True, on_step=False, on_epoch=True)
        self.log('valid_mae_epoch', valid_mae, prog_bar=True, on_step=False, on_epoch=True)
        self.log('valid_MAD_epoch', valid_ens_dict["MAD"], prog_bar=True, on_step=False, on_epoch=True)
        self.log('valid_OLS_epoch', valid_ens_dict["OLS"], prog_bar=True, on_step=False, on_epoch=True)
        self.log('valid_EMD_epoch', valid_ens_dict["EMD"], prog_bar=True, on_step=False, on_epoch=True)
        self.log('valid_SSIM_epoch', valid_ens_dict["SSIM"], prog_bar=True, on_step=False, on_epoch=True)
        self.log('valid_EarthNetScore_epoch', valid_ens_dict["EarthNetScore"], prog_bar=True, on_step=False,
                 on_epoch=True)
        self.valid_mse.reset()
        self.valid_mae.reset()
        self.valid_ens.reset()

    @property
    def test_epoch_count(self):
        if not hasattr(self, "_test_epoch_count"):
            self.reset_test_epoch_count()
        return self._test_epoch_count

    def increase_test_epoch_count(self, val=1):
        if not hasattr(self, "_test_epoch_count"):
            self.reset_test_epoch_count()
        self._test_epoch_count += val

    def reset_test_epoch_count(self):
        self._test_epoch_count = 0

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        seq = batch[..., :self.channels]
        mask = batch[..., self.channels:][self.out_slice]
        x = seq[self.in_slice]
        y = seq[self.out_slice]
        micro_batch_size = x.shape[self.batch_axis]
        data_idx = int(batch_idx * micro_batch_size)
        if not self.eval_example_only or data_idx in self.test_example_data_idx_list:
            y_hat, _ = self(batch)
            if self.local_rank == 0:
                self.save_vis_step_end(
                    data_idx=data_idx,
                    context_seq=seq[self.in_slice].detach().float().cpu().numpy(),
                    target_seq=seq[self.out_slice].detach().float().cpu().numpy(),
                    pred_seq=y_hat.detach().float().cpu().numpy(),
                    mode="test", )
            if self.precision == 16:
                y_hat = y_hat.float()
            self.test_mse(y_hat * (1 - mask), y * (1 - mask))
            self.test_mae(y_hat * (1 - mask), y * (1 - mask))
            self.test_ens(y_hat, y, mask)
        return None

    def test_epoch_end(self, outputs):
        test_mse = self.test_mse.compute()
        test_mae = self.test_mae.compute()
        test_ens_dict = self.test_ens.compute()

        prefix = self.test_subset_name[self.test_epoch_count]
        self.log(f'{prefix}_test_mse_epoch', test_mse, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f'{prefix}_test_mae_epoch', test_mae, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f'{prefix}_test_MAD_epoch', test_ens_dict["MAD"], prog_bar=True, on_step=False, on_epoch=True)
        self.log(f'{prefix}_test_OLS_epoch', test_ens_dict["OLS"], prog_bar=True, on_step=False, on_epoch=True)
        self.log(f'{prefix}_test_EMD_epoch', test_ens_dict["EMD"], prog_bar=True, on_step=False, on_epoch=True)
        self.log(f'{prefix}_test_SSIM_epoch', test_ens_dict["SSIM"], prog_bar=True, on_step=False, on_epoch=True)
        self.log(f'{prefix}_test_EarthNetScore_epoch', test_ens_dict["EarthNetScore"], prog_bar=True, on_step=False,
                 on_epoch=True)
        self.test_mse.reset()
        self.test_mae.reset()
        self.test_ens.reset()

        self.increase_test_epoch_count()

    def save_vis_step_end(
            self,
            data_idx: int,
            context_seq: np.ndarray,
            target_seq: np.ndarray,
            pred_seq: np.ndarray,
            mode: str = "train", ):
        r"""
        Parameters
        ----------
        data_idx
        context_seq, target_seq, pred_seq:   np.ndarray
            layout should not include batch
        mode:   str
        """
        if self.local_rank == 0:
            if mode == "train":
                example_data_idx_list = self.train_example_data_idx_list
            elif mode == "val":
                example_data_idx_list = self.val_example_data_idx_list
            elif mode == "test":
                example_data_idx_list = self.test_example_data_idx_list
            else:
                raise ValueError(f"Wrong mode {mode}! Must be in ['train', 'val', 'test'].")
            if data_idx in example_data_idx_list:
                for variable in ["rgb", "ndvi"]:
                    save_name = f"{mode}_epoch_{self.current_epoch}_{variable}_data_{data_idx}.png"
                    vis_earthnet_seq(
                        context_np=context_seq,
                        target_np=target_seq,
                        pred_np=pred_seq,
                        ncols=self.oc.vis.ncols,
                        layout=self.layout,
                        variable=variable,
                        vegetation_mask=None,
                        cloud_mask=True,
                        save_path=os.path.join(self.example_save_dir, save_name),
                        dpi=self.oc.vis.dpi,
                        figsize=self.oc.vis.figsize,
                        font_size=self.oc.vis.font_size,
                        y_label_rotation=self.oc.vis.y_label_rotation,
                        y_label_offset=self.oc.vis.y_label_offset, )

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default='tmp_earthnet2021', type=str)
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--cfg', default=None, type=str)
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.cfg is not None:
        oc_from_file = OmegaConf.load(open(args.cfg, "r"))
        dataset_cfg = OmegaConf.to_object(oc_from_file.dataset)
        micro_batch_size = oc_from_file.optim.micro_batch_size
        seed = oc_from_file.optim.seed
    else:
        dataset_cfg = OmegaConf.to_object(PersistenceEarthNet2021PLModule.get_dataset_config())
        micro_batch_size = 1
        seed = 0
    seed_everything(seed, workers=True)
    dataloader_dict = PersistenceEarthNet2021PLModule.get_earthnet2021_dataloaders(
        dataset_cfg=dataset_cfg,
        micro_batch_size=micro_batch_size,
        num_workers=8, )
    pl_module = PersistenceEarthNet2021PLModule(
        save_dir=args.save,
        oc_file=args.cfg)
    trainer_kwargs = pl_module.set_trainer_kwargs(
        devices=args.gpus,
    )
    trainer = Trainer(**trainer_kwargs)
    pl_module.reset_test_epoch_count()
    for test_dataloader in dataloader_dict["test_dataloader"]:
        trainer.test(model=pl_module,
                     dataloaders=test_dataloader)

if __name__ == "__main__":
    main()
