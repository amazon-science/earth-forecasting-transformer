import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning import loggers as pl_loggers
import logging
import wandb

from shutil import copyfile
import numpy as np

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor
from src.earthformer.utils.apex_ddp import ApexDDPStrategy

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.nn import functional as F
import torchmetrics

from src.earthformer.datasets.ims.ims_datamodule import IMSLightningDataModule
from src.earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel
from src.earthformer.visualization.ims.ims_visualize import IMSVisualize

from src.earthformer.utils.optim import SequentialLR, warmup_lambda

from datetime import datetime
from omegaconf import OmegaConf
import os
import argparse


class CuboidIMSModule(pl.LightningModule):

    def __init__(self,
                 cfg_file_path: str = None,
                 logging_dir: str = None):
        super(CuboidIMSModule, self).__init__()

        if cfg_file_path is None:
            self.cfg_file_path = os.path.join(os.path.dirname(__file__), "cfg_ims.yaml")
        else:
            self.cfg_file_path = cfg_file_path

        # save hyperparams
        train_cfg = OmegaConf.load(open(self.cfg_file_path, "r"))
        self.save_hyperparameters(train_cfg)

        # data module
        self.dm = self._get_dm()

        # torch nn module
        self.torch_nn_module = self._get_torch_nn_module()

        self.train_loss = F.mse_loss
        self.validation_loss = torchmetrics.MeanSquaredError()  # TODO: why they are different?

        # total_num_steps = (number of epochs) * (number of batches in the train data)
        self.total_num_steps = int(self.hparams.optim.max_epochs *
                                   len(self.dm.ims_train) /
                                   self.hparams.optim.total_batch_size)

        # create logging directories and set up logging
        self._init_logging(logging_dir)

    def _init_logging(self, logging_dir: str = None):
        # creates logging directories and adds their path as data members
        if logging_dir is None:
            logging_dir = os.path.join(os.path.dirname(__file__), "logging")
        self.logging_dir = logging_dir
        os.makedirs(self.logging_dir, exist_ok=True)
        
        self.our_logs_dir = os.path.join(self.logging_dir, "our_logs")
        os.makedirs(self.our_logs_dir, exist_ok=True)
                
        # add a new directory for the curr version 
        max_version_num = self.hparams.logging.first_version_num
        for f in os.listdir(self.our_logs):
            if os.path.isdir(f):
                if (f.split("_")[-1]).isnumeric():
                    max_version_num = max(max_version_num, int(f.split("_")[-1]))

        self.curr_version_dir = os.path.join(self.our_logs_dir, str(max_version_num))
        os.makedirs(self.curr_version_dir, exist_ok=True)

        self.scores_dir = os.path.join(self.curr_version_dir, "scores")
        self.examples_dir = os.path.join(self.curr_version_dir, "examples")
        self.checkpoints_dir = os.path.join(self.curr_version_dir, "checkpoints")
        os.makedirs(self.scores_dir, exist_ok=True)
        os.makedirs(self.examples_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # get visualization config
        self.visualize = IMSVisualize(save_dir=self.examples_dir,
                                      scale=self.hparams.dataset.preprocess.scale,
                                      fs=self.hparams.logging.visualize.fs,
                                      figsize=self.hparams.logging.visualize.figsize,
                                      plot_stride=self.hparams.logging.visualize.plot_stride)

        # save a copy of the current config inside the logging dir
        cfg_file_target_path = os.path.join(self.curr_version_dir, "cfg.yaml")
        if (not os.path.exists(cfg_file_target_path)) or \
                (not os.path.samefile(self.cfg_file_path, cfg_file_target_path)):
            copyfile(self.cfg_file_path, cfg_file_target_path)

    def _get_x_y_from_batch(self, batch):
        # batch.shape is (N, T, H, W, C)
        return batch[:, :self.hparams.model.in_len, :, :, :], \
               batch[:, self.hparams.model.in_len:(self.hparams.model.in_len + self.hparams.model.out_len), :, :, :]

    def forward(self, x):
        return self.torch_nn_module(x)

    def training_step(self, batch, batch_idx):
        x, y = self._get_x_y_from_batch(batch)
        y_hat = self(x)
        loss = self.train_loss(y_hat, y)

        data_idx = int(batch_idx * self.hparams.optim.micro_batch_size)

        # save our visualization
        self.save_visualization(in_seq=x[0],
                                target_seq=y[0],
                                pred_seq_list=y_hat[0],
                                data_idx=data_idx,
                                mode="train")

        self.log('train_loss_step', loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def predict_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        if self.hparams.optim.method == 'adamw':
            optimizer = AdamW(params=self.parameters(),
                              lr=self.hparams.optim.lr,
                              weight_decay=self.hparams.optim.wd)
        else:
            raise NotImplementedError

        warmup_iter = int(np.round(self.hparams.optim.warmup_percentage * self.total_num_steps))

        if self.hparams.optim.lr_scheduler_mode == 'cosine':
            warmup_scheduler = LambdaLR(optimizer,
                                        lr_lambda=warmup_lambda(warmup_steps=warmup_iter,
                                                                min_lr_ratio=self.hparams.optim.warmup_min_lr_ratio))
            cosine_scheduler = CosineAnnealingLR(optimizer,
                                                 T_max=(self.total_num_steps - warmup_iter),
                                                 eta_min=self.hparams.optim.min_lr_ratio * self.hparams.optim.lr)
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

    def get_trainer_kwargs(self, gpus):
        r"""
        Default kwargs used when initializing pl.Trainer
        """
        # TODO: early stopping not implemented currently
        # because it depends on SEVIRSkillScore objects
        # if self.hparams.optim.early_stop:
        #     callbacks += [EarlyStopping(monitor="valid_loss_epoch",
        #                                 min_delta=0.0,
        #                                 patience=self.oc.optim.early_stop_patience,
        #                                 verbose=False,
        #                                 mode=self.oc.optim.early_stop_mode), ]

        # ModelCheckpoint allows fine-grained control over checkpointing
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss_epoch",
            filename="model-{epoch:03d}",
            save_top_k=self.hparams.optim.save_top_k,
            save_last=True,
            mode="min",
            dirpath=self.checkpoints_dir
        )

        callbacks = []
        callbacks += [checkpoint_callback, ]
        if self.hparams.logging.monitor_lr:
            callbacks += [LearningRateMonitor(logging_interval='step'), ]
        if self.hparams.logging.monitor_device:
            callbacks += [DeviceStatsMonitor(), ]

        # logging
        loggers = []
        """ 
        TensorBoardLogger - to see the metrics run the following commands on timon:
        cd leah/cloud-forecasting-transformer/
        tensorboard --logdir scripts/cuboid_transformer/ims/logging/lightning_logs --bind_all
        
        Then, go to chrome and connect http://http://192.168.0.177/:6006/.
        """
        if self.hparams.logging.use_tensorbaord:
            loggers.append(pl_loggers.TensorBoardLogger(save_dir=self.logging_dir))

        """
        CSVLogger
        """
        if self.hparams.logging.use_csv:
            loggers.append(pl_loggers.CSVLogger(save_dir=self.logging_dir))

        """
        WandbLogger
        """
        if self.hparams.logging.use_wandb:
            loggers.append(pl_loggers.WandbLogger(project="cloud-forecasting-transformer", save_dir=self.logging_dir))

        trainer_kwargs = dict(
            devices=gpus,
            accumulate_grad_batches=self.hparams.optim.total_batch_size // (self.hparams.optim.micro_batch_size * gpus),
            callbacks=callbacks,
            # log
            logger=loggers,
            log_every_n_steps=max(1, int(self.hparams.trainer.log_step_ratio * self.total_num_steps)),
            track_grad_norm=self.hparams.logging.track_grad_norm,
            # save
            default_root_dir=self.logging_dir,
            # ddp
            accelerator=self.hparams.trainer.accelerator,
            strategy=ApexDDPStrategy(find_unused_parameters=False, delay_allreduce=True),
            # optimization
            max_epochs=self.hparams.optim.max_epochs,
            check_val_every_n_epoch=self.hparams.trainer.check_val_every_n_epoch,
            gradient_clip_val=self.hparams.optim.gradient_clip_val,
            # NVIDIA amp
            precision=self.hparams.trainer.precision,
        )
        return trainer_kwargs

    def validation_step(self, batch, batch_idx):
        x, y = self._get_x_y_from_batch(batch)
        y_hat = self(x)

        # TODO: verify we know what it means, seems like the first in any microbatch
        data_idx = int(
            batch_idx * self.hparams.optim.micro_batch_size)

        # save our visualization
        self.save_visualization(in_seq=x[0],
                                target_seq=y[0],
                                pred_seq_list=y_hat[0],
                                data_idx=data_idx,
                                mode="val")

        loss = self.validation_loss(y_hat, y)
        self.log('val_loss_step', loss, prog_bar=True, on_step=True, on_epoch=False)

    def validation_epoch_end(self, outputs):
        epoch_loss = self.validation_loss.compute()
        self.log("val_loss_epoch", epoch_loss, sync_dist=True, on_epoch=True)
        self.validation_loss.reset()

    def save_visualization(
            self,
            data_idx: int,
            in_seq: torch.Tensor,
            target_seq: torch.Tensor,
            pred_seq_list: torch.Tensor,
            mode: str = "train"):

        if mode == "train":
            example_data_idx_list = self.hparams.logging.visualize.train_example_data_idx_list
        elif mode == "val":
            example_data_idx_list = self.hparams.logging.visualize.val_example_data_idx_list
        elif mode == "test":
            example_data_idx_list = self.hparams.logging.visualize.test_example_data_idx_list
        else:
            raise ValueError(f"Wrong mode {mode}! Must be in ['train', 'val', 'test'].")

        in_seq = in_seq.detach().float().cpu().numpy()
        target_seq = target_seq.detach().float().cpu().numpy()
        pred_seq_list = pred_seq_list.detach().float().cpu().numpy()

        if data_idx in example_data_idx_list:
            self.visualize.save_example(save_prefix=f'{mode}_epoch_{self.current_epoch}_data_{data_idx}',
                                        in_seq=in_seq,
                                        target_seq=target_seq,
                                        pred_seq_list=[pred_seq_list],
                                        time_delta=self.hparams.dataset.time_delta,
                                        )

            if self.hparams.logging.use_wandb:
                x_images = [wandb.Image(image) for image in in_seq]
                y_images = [wandb.Image(image) for image in target_seq]
                y_hat_images = [wandb.Image(image) for image in pred_seq_list]

                wandb.log({f"{mode}": {"x": x_images, "y": y_images, "y_hat": y_hat_images}})

    def _get_dm(self):
        dm = IMSLightningDataModule(start_date=datetime(*self.hparams.dataset.start_date),
                                    # TODO: get date filter for each one instead of a fixed date
                                    train_val_split_date=datetime(*self.hparams.dataset.train_val_split_date),
                                    train_test_split_date=datetime(*self.hparams.dataset.train_test_split_date),
                                    end_date=datetime(*self.hparams.dataset.end_date),
                                    batch_size=self.hparams.optim.micro_batch_size,
                                    batch_layout=self.hparams.dataset.batch_layout,
                                    num_workers=self.hparams.optim.num_workers,
                                    img_type=self.hparams.dataset.img_type,
                                    seq_len=self.hparams.dataset.seq_len,
                                    stride=self.hparams.dataset.stride,
                                    time_delta=self.hparams.dataset.time_delta,
                                    layout=self.hparams.dataset.layout,
                                    ims_catalog=self.hparams.dataset.ims_catalog,
                                    ims_data_dir=self.hparams.dataset.ims_data_dir,
                                    grayscale=self.hparams.dataset.preprocess.grayscale,
                                    left=self.hparams.dataset.preprocess.crop.left,
                                    top=self.hparams.dataset.preprocess.crop.top,
                                    width=self.hparams.dataset.preprocess.crop.width,
                                    height=self.hparams.dataset.preprocess.crop.height,
                                    scale=self.hparams.dataset.preprocess.scale,
                                    )
        dm.prepare_data()
        dm.setup()
        return dm

    def _get_torch_nn_module(self):
        model_cfg = self.hparams.model

        # ---- compute fields that require arithmetic operations on config values  ---- #
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
        return CuboidTransformerModel(
            # --------------------------- network arch/size --------------------------- #
            # --- general encoder/decoder configs
            enc_depth=model_cfg["enc_depth"],
            dec_depth=model_cfg["dec_depth"],
            enc_use_inter_ffn=model_cfg["enc_use_inter_ffn"],
            dec_use_inter_ffn=model_cfg["dec_use_inter_ffn"],
            # --- attention related
            enc_attn_patterns=enc_attn_patterns,
            dec_self_attn_patterns=dec_self_attn_patterns,
            dec_cross_attn_patterns=dec_cross_attn_patterns,
            dec_cross_last_n_frames=model_cfg["dec_cross_last_n_frames"],
            dec_use_first_self_attn=model_cfg["dec_use_first_self_attn"],
            num_heads=model_cfg["num_heads"],
            attn_drop=model_cfg["attn_drop"],
            dec_hierarchical_pos_embed=model_cfg["dec_hierarchical_pos_embed"],
            pos_embed_type=model_cfg["pos_embed_type"],
            use_relative_pos=model_cfg["use_relative_pos"],
            self_attn_use_final_proj=model_cfg["self_attn_use_final_proj"],
            # --- in/out shape
            input_shape=[model_cfg["in_len"], *self.dm.get_hwc()],
            target_shape=[model_cfg["out_len"], *self.dm.get_hwc()],
            padding_type=model_cfg["padding_type"],
            # --- TODO: what are these?
            base_units=model_cfg["base_units"],
            block_units=model_cfg["block_units"],  # this is null in cfg_ims.yaml
            scale_alpha=model_cfg["scale_alpha"],  # not necessary ?
            # --- voodoo stuff that hopefully helps so everyone do it
            proj_drop=model_cfg["proj_drop"],
            ffn_drop=model_cfg["ffn_drop"],
            upsample_type=model_cfg["upsample_type"],
            downsample=model_cfg["downsample"],
            downsample_type=model_cfg["downsample_type"],
            ffn_activation=model_cfg["ffn_activation"],
            gated_ffn=model_cfg["gated_ffn"],
            norm_layer=model_cfg["norm_layer"],
            # --- initial_downsample
            initial_downsample_type=model_cfg["initial_downsample_type"],
            initial_downsample_activation=model_cfg["initial_downsample_activation"],
            # these are relevant when (initial_downsample_type == "stack_conv")
            initial_downsample_stack_conv_num_layers=model_cfg["initial_downsample_stack_conv_num_layers"],
            initial_downsample_stack_conv_dim_list=model_cfg["initial_downsample_stack_conv_dim_list"],
            initial_downsample_stack_conv_downscale_list=model_cfg["initial_downsample_stack_conv_downscale_list"],
            initial_downsample_stack_conv_num_conv_list=model_cfg["initial_downsample_stack_conv_num_conv_list"],
            # --- global vectors
            num_global_vectors=model_cfg["num_global_vectors"],
            use_dec_self_global=model_cfg["use_dec_self_global"],
            dec_self_update_global=model_cfg["dec_self_update_global"],
            use_dec_cross_global=model_cfg["use_dec_cross_global"],
            use_global_vector_ffn=model_cfg["use_global_vector_ffn"],
            use_global_self_attn=model_cfg["use_global_self_attn"],
            separate_global_qkv=model_cfg["separate_global_qkv"],
            global_dim_ratio=model_cfg["global_dim_ratio"],
            # ----------------------------- initialization ---------------------------- #
            attn_linear_init_mode=model_cfg["attn_linear_init_mode"],
            ffn_linear_init_mode=model_cfg["ffn_linear_init_mode"],
            conv_init_mode=model_cfg["conv_init_mode"],
            down_up_linear_init_mode=model_cfg["down_up_linear_init_mode"],
            norm_init_mode=model_cfg["norm_init_mode"],
            # ----------------------------------- misc -------------------------------- #
            z_init_method=model_cfg["z_init_method"],
            checkpoint_level=model_cfg["checkpoint_level"],
        )


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logging-dir', default=None, type=str)
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--cfg', default=None, type=str)
    # TODO: return to these arguments when they will be relevant
    # parser.add_argument('--test', action='store_true')
    # parser.add_argument('--pretrained', action='store_true',
    #                     help='Load pretrained checkpoints for test.')
    # parser.add_argument('--ckpt-name', default=None, type=str,
    #                     help='The model checkpoint trained on IMS.')
    return parser


def main():
    logging.getLogger("lightning").setLevel(logging.ERROR)  # suppress WARN massages in console

    parser = get_parser()
    args = parser.parse_args()

    # TODO: start from a saved checkpoint with l_model.load_from_checkpoint(PATH)
    # TODO: config optimizer
    # TODO: test from a pretrained checkpoint like sevir did

    # model
    l_model = CuboidIMSModule(logging_dir=args.logging_dir,
                              cfg_file_path=args.cfg)
    # data
    dm = l_model.dm

    # seed
    seed_everything(seed=l_model.hparams.optim.seed, workers=True)

    # train model
    trainer_kwargs = l_model.get_trainer_kwargs(args.gpus)
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model=l_model, datamodule=dm)


if __name__ == "__main__":
    main()
