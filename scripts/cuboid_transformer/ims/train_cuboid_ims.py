import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from torch.optim import AdamW
from torch.nn import functional as F
import torchmetrics

from src.earthformer.datasets.ims.ims_datamodule import IMSLightningDataModule
from src.earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel

from datetime import datetime
from omegaconf import OmegaConf
import os


class CuboidIMSModule(pl.LightningModule):

    def __init__(self,
                 total_num_steps: int = 10,
                 cfg_file_path: str = None,
                 logging_dir: str = None):
        super(CuboidIMSModule, self).__init__()

        assert cfg_file_path != None, "Error: config file must be provided to CuboidIMSModule"
        self.cfg_file_path = cfg_file_path

        if logging_dir is None:
            self.logging_dir = os.path.join(os.path.dirname(__file__), "logging") # TODO: create if not present
        else:
            self.logging_dir = logging_dir
        assert total_num_steps > 0, "Error: total_num_steps have to be positive!"
        self.total_num_steps = total_num_steps

        train_cfg = OmegaConf.load(open(cfg_file_path, "r"))
        model_cfg = train_cfg.model

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
        self.torch_nn_module = CuboidTransformerModel(
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
            input_shape=model_cfg["input_shape"],
            target_shape=model_cfg["target_shape"],
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

        self.save_hyperparameters(train_cfg)
        self.init_data_members_using_conf(train_cfg, total_num_steps, logging_dir) # is this necessary?

    def init_data_members_using_conf(self, oc, total_num_steps, save_dir):
        pass
        # self.valid_mse = torchmetrics.MeanSquaredError()
        # self.valid_mae = torchmetrics.MeanAbsoluteError()
        # self.valid_score = SEVIRSkillScore(
        #     mode=self.metrics_mode,
        #     seq_len=self.out_len,
        #     layout=self.layout,
        #     threshold_list=self.threshold_list,
        #     metrics_list=self.metrics_list,
        #     eps=1e-4,)
        # self.test_mse = torchmetrics.MeanSquaredError()
        # self.test_mae = torchmetrics.MeanAbsoluteError()
        # self.test_score = SEVIRSkillScore(
        #     mode=self.metrics_mode,
        #     seq_len=self.out_len,
        #     layout=self.layout,
        #     threshold_list=self.threshold_list,
        #     metrics_list=self.metrics_list,
        #     eps=1e-4,)

    def forward(self, x):
        return self.torch_nn_module(x)

    def training_step(self, batch, batch_idx):
        # batch.shape is (N, T, H, W, C)
        x, y = batch[:, :self.hparams.layout.in_len, :, :, :], batch[:, self.hparams.layout.in_len:(self.hparams.layout.in_len + self.hparams.layout.out_len), :, :, :]
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y) # TODO: we did not use the self.mse attributes.

        # TODO: check the logging
        self.log('train_loss', loss,
                 on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def predict_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters())  # TODO: give params
        return optimizer

    def get_dm(self):
        dm = IMSLightningDataModule(start_date=datetime(*self.hparams.dataset.start_date),
                               # TODO: get date filter for each one instead of a fixed date
                               train_val_split_date=datetime(*self.hparams.dataset.train_val_split_date),
                               train_test_split_date=datetime(*self.hparams.dataset.train_test_split_date),
                               end_date=datetime(*self.hparams.dataset.end_date),
                               batch_size=self.hparams.optim.micro_batch_size,
                               batch_layout=self.hparams.layout.batch_layout,
                               num_workers=self.hparams.optim.num_workers,
                               img_type=self.hparams.dataset.img_type,
                               seq_len=self.hparams.dataset.seq_len,
                               stride=self.hparams.dataset.stride,
                               layout=self.hparams.dataset.layout,
                               ims_catalog=self.hparams.dataset.ims_catalog,
                               ims_data_dir=self.hparams.dataset.ims_data_dir,
                               # TODO: addd preprocess to cfg
                               )
        dm.prepare_data()
        dm.setup()
        return dm


def main():
    # TODO: get command line args! (cfg file path for example)
    cfg_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "cfg_ims.yaml"))

    # model
    l_model = CuboidIMSModule(cfg_file_path=cfg_file_path)

    # data
    dm = l_model.get_dm()

    # train model
    trainer = pl.Trainer(max_epochs=l_model.hparams.optim.max_epochs)
    trainer.fit(model=l_model, train_dataloaders=dm.train_dataloader())

if __name__ == "__main__":
    main()
