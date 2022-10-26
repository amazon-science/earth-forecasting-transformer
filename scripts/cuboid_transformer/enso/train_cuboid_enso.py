import warnings
from typing import Sequence
from shutil import copyfile
import inspect
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything, loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor, Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from omegaconf import OmegaConf
import os
import argparse
from einops import rearrange
from earthformer.config import cfg
from earthformer.utils.optim import SequentialLR, warmup_lambda
from earthformer.utils.utils import get_parameter_names
from earthformer.utils.checkpoint import pl_ckpt_to_pytorch_state_dict, s3_download_pretrained_ckpt
from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel
from earthformer.datasets.enso.enso_dataloader import ENSOLightningDataModule, NINO_WINDOW_T
from earthformer.metrics.enso import sst_to_nino, compute_enso_score
from earthformer.utils.apex_ddp import ApexDDPPlugin


pytorch_state_dict_name = "earthformer_icarenso2021.pt"

class CuboidENSOPLModule(pl.LightningModule):

    def __init__(self,
                 total_num_steps: int,
                 oc_file: str = None,
                 save_dir: str = None):
        super(CuboidENSOPLModule, self).__init__()
        if oc_file is not None:
            oc_from_file = OmegaConf.load(open(oc_file, "r"))
        else:
            oc_from_file = None
        oc = self.get_base_config(oc_from_file=oc_from_file)
        model_cfg = OmegaConf.to_object(oc.model)
        num_blocks = len(model_cfg["enc_depth"])
        if isinstance(model_cfg["self_pattern"], str):
            enc_attn_patterns = [model_cfg["self_pattern"]] * num_blocks
        else:
            enc_attn_patterns = OmegaConf.to_container(model_cfg["self_pattern"])
        if isinstance(model_cfg["cross_self_pattern"], str):
            dec_self_attn_patterns = [model_cfg["cross_self_pattern"]] * num_blocks
        else:
            dec_self_attn_patterns = OmegaConf.to_container(model_cfg["cross_self_pattern"])
        if isinstance(model_cfg["cross_pattern"], str):
            dec_cross_attn_patterns = [model_cfg["cross_pattern"]] * num_blocks
        else:
            dec_cross_attn_patterns = OmegaConf.to_container(model_cfg["cross_pattern"])

        self.torch_nn_module = CuboidTransformerModel(
            input_shape=model_cfg["input_shape"],
            target_shape=model_cfg["target_shape"],
            base_units=model_cfg["base_units"],
            block_units=model_cfg["block_units"],
            scale_alpha=model_cfg["scale_alpha"],
            enc_depth=model_cfg["enc_depth"],
            dec_depth=model_cfg["dec_depth"],
            enc_use_inter_ffn=model_cfg["enc_use_inter_ffn"],
            dec_use_inter_ffn=model_cfg["dec_use_inter_ffn"],
            downsample=model_cfg["downsample"],
            downsample_type=model_cfg["downsample_type"],
            enc_attn_patterns=enc_attn_patterns,
            dec_self_attn_patterns=dec_self_attn_patterns,
            dec_cross_attn_patterns=dec_cross_attn_patterns,
            dec_cross_last_n_frames=model_cfg["cross_last_n_frames"],
            dec_use_first_self_attn=model_cfg["dec_use_first_self_attn"],
            num_heads=model_cfg["num_heads"],
            attn_drop=model_cfg["attn_drop"],
            proj_drop=model_cfg["proj_drop"],
            ffn_drop=model_cfg["ffn_drop"],
            upsample_type=model_cfg["upsample_type"],
            ffn_activation=model_cfg["ffn_activation"],
            gated_ffn=model_cfg["gated_ffn"],
            norm_layer=model_cfg["norm_layer"],
            # global vectors
            num_global_vectors=model_cfg["num_global_vectors"],
            use_dec_self_global=model_cfg["use_dec_self_global"],
            dec_self_update_global=model_cfg["dec_self_update_global"],
            use_dec_cross_global=model_cfg["use_dec_cross_global"],
            use_global_vector_ffn=model_cfg["use_global_vector_ffn"],
            use_global_self_attn=model_cfg["use_global_self_attn"],
            separate_global_qkv=model_cfg["separate_global_qkv"],
            global_dim_ratio=model_cfg["global_dim_ratio"],
            # initial_downsample
            initial_downsample_type=model_cfg["initial_downsample_type"],
            initial_downsample_activation=model_cfg["initial_downsample_activation"],
            # initial_downsample_type=="conv"
            initial_downsample_scale=model_cfg["initial_downsample_scale"],
            initial_downsample_conv_layers=model_cfg["initial_final_sample_num_conv"],
            final_upsample_conv_layers=model_cfg["initial_final_sample_num_conv"] - 1,
            # misc
            padding_type=model_cfg["padding_type"],
            z_init_method=model_cfg["z_init_method"],
            checkpoint_level=model_cfg["checkpoint_level"],
            pos_embed_type=model_cfg["pos_embed_type"],
            use_relative_pos=model_cfg["use_relative_pos"],
            self_attn_use_final_proj=model_cfg["self_attn_use_final_proj"],
            # initialization
            attn_linear_init_mode=model_cfg["attn_linear_init_mode"],
            ffn_linear_init_mode=model_cfg["ffn_linear_init_mode"],
            conv_init_mode=model_cfg["conv_init_mode"],
            down_up_linear_init_mode=model_cfg["down_up_linear_init_mode"],
            norm_init_mode=model_cfg["norm_init_mode"],
        )

        self.total_num_steps = total_num_steps
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
        self.channels = model_cfg["data_channels"]
        # dataset
        self.normalize_sst = oc.dataset.normalize_sst
        # optimization
        self.max_epochs = oc.optim.max_epochs
        self.optim_method = oc.optim.method
        self.lr = oc.optim.lr
        self.wd = oc.optim.wd
        # lr_scheduler
        self.total_num_steps = total_num_steps
        self.lr_scheduler_mode = oc.optim.lr_scheduler_mode
        self.warmup_percentage = oc.optim.warmup_percentage
        self.min_lr_ratio = oc.optim.min_lr_ratio
        # logging
        self.save_dir = save_dir
        self.logging_prefix = oc.logging.logging_prefix
        # visualization
        self.train_example_data_idx_list = list(oc.vis.train_example_data_idx_list)
        self.val_example_data_idx_list = list(oc.vis.val_example_data_idx_list)
        self.test_example_data_idx_list = list(oc.vis.test_example_data_idx_list)
        self.eval_example_only = oc.vis.eval_example_only

        self.valid_mse = torchmetrics.MeanSquaredError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        self.test_mse = torchmetrics.MeanSquaredError()
        self.test_mae = torchmetrics.MeanAbsoluteError()

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
        oc.trainer = self.get_trainer_config()
        oc.vis = self.get_vis_config()
        oc.model = self.get_model_config()
        oc.dataset = self.get_dataset_config()
        if oc_from_file is not None:
            # oc = apply_omegaconf_overrides(oc, oc_from_file)
            oc = OmegaConf.merge(oc, oc_from_file)
        return oc

    @staticmethod
    def get_layout_config():
        cfg = OmegaConf.create()
        cfg.in_len = 12
        cfg.out_len = 14
        cfg.img_height = 24
        cfg.img_width = 48
        cfg.layout = "NTHWC"
        return cfg

    @classmethod
    def get_model_config(cls):
        cfg = OmegaConf.create()
        layout_cfg = cls.get_layout_config()
        cfg.data_channels = 4
        cfg.input_shape = (layout_cfg.in_len, layout_cfg.img_height, layout_cfg.img_width, cfg.data_channels)
        cfg.target_shape = (layout_cfg.out_len, layout_cfg.img_height, layout_cfg.img_width, cfg.data_channels)

        cfg.base_units = 64
        cfg.block_units = None # multiply by 2 when downsampling in each layer
        cfg.scale_alpha = 1.0

        cfg.enc_depth = [1, 1]
        cfg.dec_depth = [1, 1]
        cfg.enc_use_inter_ffn = True
        cfg.dec_use_inter_ffn = True
        cfg.dec_hierarchical_pos_embed = True

        cfg.downsample = 2
        cfg.downsample_type = "patch_merge"
        cfg.upsample_type = "upsample"

        cfg.num_global_vectors = 8
        cfg.use_dec_self_global = True
        cfg.dec_self_update_global = True
        cfg.use_dec_cross_global = True
        cfg.use_global_vector_ffn = True
        cfg.use_global_self_attn = False
        cfg.separate_global_qkv = False
        cfg.global_dim_ratio = 1

        cfg.self_pattern = 'axial'
        cfg.cross_self_pattern = 'axial'
        cfg.cross_pattern = 'cross_1x1'
        cfg.cross_last_n_frames = None

        cfg.attn_drop = 0.1
        cfg.proj_drop = 0.1
        cfg.ffn_drop = 0.1
        cfg.num_heads = 4

        cfg.ffn_activation = 'gelu'
        cfg.gated_ffn = False
        cfg.norm_layer = 'layer_norm'
        cfg.padding_type = 'zeros'
        cfg.pos_embed_type = "t+hw"
        cfg.use_relative_pos = True
        cfg.self_attn_use_final_proj = True
        cfg.dec_use_first_self_attn = False

        cfg.z_init_method = 'zeros'  # The method for initializing the first input of the decoder
        cfg.initial_downsample_type = "conv"
        cfg.initial_downsample_activation = "leaky"
        cfg.initial_downsample_scale = (1, 1, 2)
        cfg.initial_final_sample_num_conv = 2
        cfg.checkpoint_level = 2
        # initialization
        cfg.attn_linear_init_mode = "0"
        cfg.ffn_linear_init_mode = "0"
        cfg.conv_init_mode = "0"
        cfg.down_up_linear_init_mode = "0"
        cfg.norm_init_mode = "0"
        return cfg

    @classmethod
    def get_dataset_config(cls):
        cfg = OmegaConf.create()
        cfg.in_len = 12
        cfg.out_len = 14
        cfg.nino_window_t = NINO_WINDOW_T
        cfg.in_stride = 1
        cfg.out_stride = 1
        cfg.train_samples_gap = 10
        cfg.eval_samples_gap = 1
        cfg.normalize_sst = True      
        return cfg

    @staticmethod
    def get_optim_config():
        cfg = OmegaConf.create()
        cfg.seed = None
        cfg.total_batch_size = 32
        cfg.micro_batch_size = 8

        cfg.method = "adamw"
        cfg.lr = 1E-3
        cfg.wd = 1E-5
        cfg.gradient_clip_val = 1.0
        cfg.max_epochs = 50
        # scheduler
        cfg.warmup_percentage = 0.2
        cfg.lr_scheduler_mode = "cosine"  # Can be strings like 'linear', 'cosine', 'platue'
        cfg.min_lr_ratio = 0.1
        cfg.warmup_min_lr_ratio = 0.1
        # early stopping
        cfg.early_stop = False
        cfg.early_stop_mode = "min"
        cfg.early_stop_patience = 5
        cfg.save_top_k = 1
        return cfg

    @staticmethod
    def get_logging_config():
        cfg = OmegaConf.create()
        cfg.logging_prefix = "ICAR_ENSO"
        cfg.monitor_lr = True
        cfg.monitor_device = False
        cfg.track_grad_norm = -1
        cfg.use_wandb = False
        return cfg

    @staticmethod
    def get_trainer_config():
        cfg = OmegaConf.create()
        cfg.check_val_every_n_epoch = 1
        cfg.log_step_ratio = 0.001  # Logging every 1% of the total training steps per epoch
        cfg.precision = 32
        return cfg

    @staticmethod
    def get_vis_config():
        cfg = OmegaConf.create()
        cfg.train_example_data_idx_list = [0, ]
        cfg.val_example_data_idx_list = [0, ]
        cfg.test_example_data_idx_list = [0, ]
        cfg.eval_example_only = False
        return cfg

    def configure_optimizers(self):
        # Configure the optimizer. Disable the weight decay for layer norm weights and all bias terms.
        decay_parameters = get_parameter_names(self.torch_nn_module, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [{
            'params': [p for n, p in self.torch_nn_module.named_parameters() if n in decay_parameters],
            'weight_decay': self.oc.optim.wd
        }, {
            'params': [p for n, p in self.torch_nn_module.named_parameters() if n not in decay_parameters],
            'weight_decay': 0.0
        }]

        if self.oc.optim.method == 'adamw':
            optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters,
                                          lr=self.oc.optim.lr,
                                          weight_decay=self.oc.optim.wd)
        else:
            raise NotImplementedError

        warmup_iter = int(np.round(self.oc.optim.warmup_percentage * self.total_num_steps))

        if self.oc.optim.lr_scheduler_mode == 'cosine':
            warmup_scheduler = LambdaLR(optimizer,
                                        lr_lambda=warmup_lambda(warmup_steps=warmup_iter,
                                                                min_lr_ratio=self.oc.optim.warmup_min_lr_ratio))
            cosine_scheduler = CosineAnnealingLR(optimizer,
                                                 T_max=(self.total_num_steps - warmup_iter),
                                                 eta_min=self.oc.optim.min_lr_ratio * self.oc.optim.lr)
            lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                                        milestones=[warmup_iter])
            lr_scheduler_config = {
                'scheduler': lr_scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        else:
            raise NotImplementedError
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}

    def set_trainer_kwargs(self, **kwargs):
        r"""
        Default kwargs used when initializing pl.Trainer
        """
        checkpoint_callback = ModelCheckpoint(
            monitor="valid_loss_epoch",
            dirpath=os.path.join(self.save_dir, "checkpoints"),
            filename="model-{epoch:03d}",
            save_top_k=self.oc.optim.save_top_k,
            save_last=True,
            mode="min",
        )
        callbacks = kwargs.pop("callbacks", [])
        assert isinstance(callbacks, list)
        for ele in callbacks:
            assert isinstance(ele, Callback)
        callbacks += [checkpoint_callback, ]
        if self.oc.logging.monitor_lr:
            callbacks += [LearningRateMonitor(logging_interval='step'), ]
        if self.oc.logging.monitor_device:
            callbacks += [DeviceStatsMonitor(), ]
        if self.oc.optim.early_stop:
            callbacks += [EarlyStopping(monitor="valid_loss_epoch",
                                        min_delta=0.0,
                                        patience=self.oc.optim.early_stop_patience,
                                        verbose=False,
                                        mode=self.oc.optim.early_stop_mode), ]

        logger = kwargs.pop("logger", [])
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=self.save_dir)
        csv_logger = pl_loggers.CSVLogger(save_dir=self.save_dir)
        logger += [tb_logger, csv_logger]
        if self.oc.logging.use_wandb:
            wandb_logger = pl_loggers.WandbLogger(project=self.oc.logging.logging_prefix,
                                                  save_dir=self.save_dir)
            logger += [wandb_logger, ]

        log_every_n_steps = max(1, int(self.oc.trainer.log_step_ratio * self.total_num_steps))
        trainer_init_keys = inspect.signature(Trainer).parameters.keys()
        ret = dict(
            callbacks=callbacks,
            # log
            logger=logger,
            log_every_n_steps=log_every_n_steps,
            track_grad_norm=self.oc.logging.track_grad_norm,
            # save
            default_root_dir=self.save_dir,
            # ddp
            accelerator="gpu",
            # strategy="ddp",
            strategy=ApexDDPPlugin(find_unused_parameters=False, delay_allreduce=True),
            # optimization
            max_epochs=self.oc.optim.max_epochs,
            check_val_every_n_epoch=self.oc.trainer.check_val_every_n_epoch,
            gradient_clip_val=self.oc.optim.gradient_clip_val,
            # NVIDIA amp
            precision=self.oc.trainer.precision,
        )
        oc_trainer_kwargs = OmegaConf.to_object(self.oc.trainer)
        oc_trainer_kwargs = {key: val for key, val in oc_trainer_kwargs.items() if key in trainer_init_keys}
        ret.update(oc_trainer_kwargs)
        ret.update(kwargs)
        return ret

    @classmethod
    def get_total_num_steps(
            cls,
            num_samples: int,
            total_batch_size: int,
            epoch: int = None):
        r"""
        Parameters
        ----------
        num_samples:    int
            The number of samples of the datasets. `num_samples / micro_batch_size` is the number of steps per epoch.
        total_batch_size:   int
            `total_batch_size == micro_batch_size * world_size * grad_accum`
        """
        if epoch is None:
            epoch = cls.get_optim_config().max_epochs
        return int(epoch * num_samples / total_batch_size)

    @staticmethod
    def get_enso_datamodule(
            dataset_cfg,
            micro_batch_size: int = 1,
            num_workers: int = 8):
        dm = ENSOLightningDataModule(
            in_len=dataset_cfg["in_len"],
            out_len=dataset_cfg["out_len"],
            in_stride=dataset_cfg["in_stride"],
            out_stride=dataset_cfg["out_stride"],
            train_samples_gap=dataset_cfg["train_samples_gap"],
            eval_samples_gap=dataset_cfg["eval_samples_gap"],
            normalize_sst=dataset_cfg["normalize_sst"],
            batch_size=micro_batch_size,
            num_workers=num_workers)
        return dm

    @property
    def nino_out_len(self):
        return self.out_len - NINO_WINDOW_T + 1

    def forward(self, batch):
        r"""
        sst_seq
            shape = (N, in_len+out_len, lat, lon)
        nino_target
            shape = (N, out_len+NINO_WINDOW_T-1)
        """
        sst_seq, nino_target = batch
        data_seq = sst_seq.float().unsqueeze(-1)
        in_seq = data_seq[:, :self.in_len, ...]  # (N, in_len, lat, lon, 1)
        target_seq = data_seq[:, self.in_len:, ...]  # (N, in_len, lat, lon, 1)
        pred_seq = self.torch_nn_module(in_seq)
        loss = F.mse_loss(pred_seq, target_seq)
        return pred_seq, loss, in_seq, target_seq, nino_target.float()

    def training_step(self, batch, batch_idx):
        pred_seq, loss, in_seq, target_seq, nino_target = self(batch)
        micro_batch_size = in_seq.shape[self.batch_axis]
        data_idx = int(batch_idx * micro_batch_size)
        if self.local_rank == 0:
            self.save_vis_step_end(
                data_idx=data_idx,
                context_seq=in_seq.detach().float().cpu().numpy(),
                target_seq=target_seq.detach().float().cpu().numpy(),
                pred_seq=pred_seq.detach().float().cpu().numpy(),
                mode="train", )
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        micro_batch_size = batch[0].shape[self.batch_axis]
        data_idx = int(batch_idx * micro_batch_size)
        if not self.eval_example_only or data_idx in self.val_example_data_idx_list:
            pred_seq, loss, in_seq, target_seq, nino_target = self(batch)
            if self.local_rank == 0:
                self.save_vis_step_end(
                    data_idx=data_idx,
                    context_seq=in_seq.detach().float().cpu().numpy(),
                    target_seq=target_seq.detach().float().cpu().numpy(),
                    pred_seq=pred_seq.detach().float().cpu().numpy(),
                    mode="val", )
            if self.precision == 16:
                pred_seq = pred_seq.float()
            self.valid_mse(pred_seq, target_seq)
            self.valid_mae(pred_seq, target_seq)
            nino_preds = sst_to_nino(sst=pred_seq[..., 0],
                                     normalize_sst=self.normalize_sst)
            return nino_preds, nino_target

    def validation_epoch_end(self, outputs):
        valid_mse = self.valid_mse.compute()
        valid_mae = self.valid_mae.compute()
        nino_preds_list, nino_target_list = map(list, zip(*outputs))
        nino_preds_list = torch.cat(nino_preds_list, dim=0)
        nino_target_list = torch.cat(nino_target_list, dim=0)
        valid_acc, valid_nino_rmse = compute_enso_score(
            y_pred=nino_preds_list, y_true=nino_target_list,
            acc_weight=None)
        valid_weighted_acc, _ = compute_enso_score(
            y_pred=nino_preds_list, y_true=nino_target_list,
            acc_weight="default")
        valid_acc /= self.nino_out_len
        valid_nino_rmse /= self.nino_out_len
        valid_weighted_acc /= self.nino_out_len
        valid_loss = -valid_acc

        self.log('valid_loss_epoch', valid_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('valid_mse_epoch', valid_mse, prog_bar=True, on_step=False, on_epoch=True)
        self.log('valid_mae_epoch', valid_mae, prog_bar=True, on_step=False, on_epoch=True)
        self.log('valid_corr_nino3.4_epoch', valid_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('valid_corr_nino3.4_weighted_epoch', valid_weighted_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('valid_nino_rmse_epoch', valid_nino_rmse, prog_bar=True, on_step=False, on_epoch=True)
        self.valid_mse.reset()
        self.valid_mae.reset()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        micro_batch_size = batch[0].shape[self.batch_axis]
        data_idx = int(batch_idx * micro_batch_size)
        if not self.eval_example_only or data_idx in self.test_example_data_idx_list:
            pred_seq, loss, in_seq, target_seq, nino_target = self(batch)
            if self.local_rank == 0:
                self.save_vis_step_end(
                    data_idx=data_idx,
                    context_seq=in_seq.detach().float().cpu().numpy(),
                    target_seq=target_seq.detach().float().cpu().numpy(),
                    pred_seq=pred_seq.detach().float().cpu().numpy(),
                    mode="test", )
            if self.precision == 16:
                pred_seq = pred_seq.float()
            self.test_mse(pred_seq, target_seq)
            self.test_mae(pred_seq, target_seq)
            nino_preds = sst_to_nino(sst=pred_seq[..., 0],
                                     normalize_sst=self.normalize_sst)
            return nino_preds, nino_target

    def test_epoch_end(self, outputs):
        test_mse = self.test_mse.compute()
        test_mae = self.test_mae.compute()
        nino_preds_list, nino_target_list = map(list, zip(*outputs))
        nino_preds_list = torch.cat(nino_preds_list, dim=0)
        nino_target_list = torch.cat(nino_target_list, dim=0)
        test_acc, test_nino_rmse = compute_enso_score(
            y_pred=nino_preds_list, y_true=nino_target_list,
            acc_weight=None)
        test_weighted_acc, _ = compute_enso_score(
            y_pred=nino_preds_list, y_true=nino_target_list,
            acc_weight="default")
        test_acc /= self.nino_out_len
        test_nino_rmse /= self.nino_out_len
        test_weighted_acc /= self.nino_out_len

        self.log('test_mse_epoch', test_mse, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_mae_epoch', test_mae, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_corr_nino3.4_epoch', test_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_corr_nino3.4_weighted_epoch', test_weighted_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_nino_rmse_epoch', test_nino_rmse, prog_bar=True, on_step=False, on_epoch=True)
        self.test_mse.reset()
        self.test_mae.reset()

    def save_vis_step_end(
            self,
            data_idx: int,
            context_seq: np.ndarray,
            target_seq: np.ndarray,
            pred_seq: np.ndarray,
            mode: str = "train",
            prefix: str = ""):
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
                # TODO: add visualization
                pass

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default='tmp_enso', type=str)
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--cfg', default=None, type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--pretrained', action='store_true',
                        help='Load pretrained checkpoints for test.')
    parser.add_argument('--ckpt_name', default=None, type=str,
                        help='The model checkpoint trained on ICAR-ENSO-2021.')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.cfg is not None:
        oc_from_file = OmegaConf.load(open(args.cfg, "r"))
        dataset_cfg = OmegaConf.to_object(oc_from_file.dataset)
        total_batch_size = oc_from_file.optim.total_batch_size
        micro_batch_size = oc_from_file.optim.micro_batch_size
        max_epochs = oc_from_file.optim.max_epochs
        seed = oc_from_file.optim.seed
    else:
        dataset_cfg = OmegaConf.to_object(CuboidENSOPLModule.get_dataset_config())
        micro_batch_size = 1
        total_batch_size = int(micro_batch_size * args.gpus)
        max_epochs = None
        seed = 0
    seed_everything(seed, workers=True)
    dm = CuboidENSOPLModule.get_enso_datamodule(
        dataset_cfg=dataset_cfg,
        micro_batch_size=micro_batch_size,
        num_workers=1, )
    dm.prepare_data()
    dm.setup()
    accumulate_grad_batches = total_batch_size // (micro_batch_size * args.gpus)
    total_num_steps = CuboidENSOPLModule.get_total_num_steps(
        epoch=max_epochs,
        num_samples=dm.num_train_samples,
        total_batch_size=total_batch_size,
    )
    pl_module = CuboidENSOPLModule(
        total_num_steps=total_num_steps,
        save_dir=args.save,
        oc_file=args.cfg)
    trainer_kwargs = pl_module.set_trainer_kwargs(
        devices=args.gpus,
        accumulate_grad_batches=accumulate_grad_batches,
    )
    trainer = Trainer(**trainer_kwargs)
    if args.pretrained:
        pretrained_ckpt_name = pytorch_state_dict_name
        if not os.path.exists(os.path.join(cfg.pretrained_checkpoints_dir, pretrained_ckpt_name)):
            s3_download_pretrained_ckpt(ckpt_name=pretrained_ckpt_name,
                                        save_dir=cfg.pretrained_checkpoints_dir,
                                        exist_ok=False)
        state_dict = torch.load(os.path.join(cfg.pretrained_checkpoints_dir, pretrained_ckpt_name),
                                map_location=torch.device("cpu"))
        pl_module.torch_nn_module.load_state_dict(state_dict=state_dict)
        trainer.test(model=pl_module,
                     datamodule=dm)
    elif args.test:
        assert args.ckpt_name is not None, f"args.ckpt_name is required for test!"
        ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
        trainer.test(model=pl_module,
                     datamodule=dm,
                     ckpt_path=ckpt_path)
    else:
        if args.ckpt_name is not None:
            ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
            if not os.path.exists(ckpt_path):
                warnings.warn(f"ckpt {ckpt_path} not exists! Start training from epoch 0.")
                ckpt_path = None
        else:
            ckpt_path = None
        trainer.fit(model=pl_module,
                    datamodule=dm,
                    ckpt_path=ckpt_path)
        state_dict = pl_ckpt_to_pytorch_state_dict(checkpoint_path=trainer.checkpoint_callback.best_model_path,
                                                   map_location=torch.device("cpu"),
                                                   delete_prefix_len=len("torch_nn_module."))
        torch.save(state_dict, os.path.join(pl_module.save_dir, "checkpoints", pytorch_state_dict_name))
        trainer.test(model=pl_module,
                     datamodule=dm)

if __name__ == "__main__":
    main()
