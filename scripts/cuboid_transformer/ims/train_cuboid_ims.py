import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from torch.optim import AdamW
from torch.nn import functional as F
import torchmetrics

from src.earthformer.datasets.ims.ims_datamodule import IMSLightningDataModule
from src.earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel

from datetime import datetime
from omegaconf import OmegaConf


class CuboidIMSModule(pl.LightningModule):

    def __init__(self,
                 total_num_steps: int = 10,
                 oc_file: str = None,
                 save_dir: str = None):
        super(CuboidIMSModule, self).__init__()

        assert oc_file != None, "Error: config file must be provided to CuboidIMSModule"
        oc_from_file = OmegaConf.load(open(oc_file, "r"))
        model_cfg = oc_from_file.model
                                                        
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
            dec_hierarchical_pos_embed=model_cfg["dec_hierarchical_pos_embed"], 
            # --- attention related
            enc_attn_patterns=enc_attn_patterns,
            dec_self_attn_patterns=dec_self_attn_patterns,
            dec_cross_attn_patterns=dec_cross_attn_patterns,
            dec_cross_last_n_frames=model_cfg["dec_cross_last_n_frames"],
            dec_use_first_self_attn=model_cfg["dec_use_first_self_attn"],
            num_heads=model_cfg["num_heads"],
            attn_drop=model_cfg["attn_drop"],
            # --- in/out shape
            input_shape=model_cfg["input_shape"],
            target_shape=model_cfg["target_shape"],
            # --- TODO: what are these?
            base_units=model_cfg["base_units"],
            block_units=model_cfg["block_units"], # this is null in cfg_ims.yaml
            scale_alpha=model_cfg["scale_alpha"], # not necessary ? 
            # --- voodoo stuff that hopefully helps so everyone do it
            proj_drop=model_cfg["proj_drop"],
            ffn_drop=model_cfg["ffn_drop"],
            upsample_type=model_cfg["upsample_type"],
            downsample=model_cfg["downsample"],  
            downsample_type=model_cfg["downsample_type"],
            ffn_activation=model_cfg["ffn_activation"],
            gated_ffn=model_cfg["gated_ffn"],
            norm_layer=model_cfg["norm_layer"],
            # --- global vectors
            num_global_vectors=model_cfg["num_global_vectors"],
            use_dec_self_global=model_cfg["use_dec_self_global"],
            dec_self_update_global=model_cfg["dec_self_update_global"],
            use_dec_cross_global=model_cfg["use_dec_cross_global"],
            use_global_vector_ffn=model_cfg["use_global_vector_ffn"],
            use_global_self_attn=model_cfg["use_global_self_attn"],
            separate_global_qkv=model_cfg["separate_global_qkv"],
            global_dim_ratio=model_cfg["global_dim_ratio"],
            # --- initial_downsample 
            initial_downsample_type=model_cfg["initial_downsample_type"],
            initial_downsample_activation=model_cfg["initial_downsample_activation"],
            # these are relevant when (initial_downsample_type == "stack_conv")
            initial_downsample_stack_conv_num_layers=model_cfg["initial_downsample_stack_conv_num_layers"],
            initial_downsample_stack_conv_dim_list=model_cfg["initial_downsample_stack_conv_dim_list"],
            initial_downsample_stack_conv_downscale_list=model_cfg["initial_downsample_stack_conv_downscale_list"],
            initial_downsample_stack_conv_num_conv_list=model_cfg["initial_downsample_stack_conv_num_conv_list"],
            # ----------------------------------- misc -------------------------------- #
            padding_type=model_cfg["padding_type"],
            z_init_method=model_cfg["z_init_method"],
            checkpoint_level=model_cfg["checkpoint_level"],
            pos_embed_type=model_cfg["pos_embed_type"],
            use_relative_pos=model_cfg["use_relative_pos"],
            self_attn_use_final_proj=model_cfg["self_attn_use_final_proj"],
            # ----------------------------- initialization ---------------------------- #
            attn_linear_init_mode=model_cfg["attn_linear_init_mode"],
            ffn_linear_init_mode=model_cfg["ffn_linear_init_mode"],
            conv_init_mode=model_cfg["conv_init_mode"],
            down_up_linear_init_mode=model_cfg["down_up_linear_init_mode"],
            norm_init_mode=model_cfg["norm_init_mode"],
        )

        self.save_hyperparameters(oc_from_file)
        self.init_data_members_using_conf(oc_from_file, total_num_steps, save_dir)
        

    def init_data_members_using_conf(self, oc, total_num_steps, save_dir):
        self.oc = oc
        # layout
        self.in_len = oc.layout.in_len
        self.out_len = oc.layout.out_len
        self.layout = oc.layout.layout
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
        # evaluation
        self.metrics_list = oc.dataset.metrics_list
        self.threshold_list = oc.dataset.threshold_list
        self.metrics_mode = oc.dataset.metrics_mode
        self.valid_mse = torchmetrics.MeanSquaredError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        # self.valid_score = SEVIRSkillScore(
        #     mode=self.metrics_mode,
        #     seq_len=self.out_len,
        #     layout=self.layout,
        #     threshold_list=self.threshold_list,
        #     metrics_list=self.metrics_list,
        #     eps=1e-4,)
        self.test_mse = torchmetrics.MeanSquaredError()
        self.test_mae = torchmetrics.MeanAbsoluteError()
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
        x, y = batch[:, :self.in_len, :, :, :], batch[:, self.in_len:(self.in_len+self.out_len), :, :, :]
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        # TODO: check the loging
        self.log('train_loss', loss,
                 on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def predict_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters()) # TODO: give params
        return optimizer


def main():
    # model
    l_model = CuboidIMSModule("/content/cloud-forecasting-transformer/scripts/cuboid_transformer/ims/cfg_ims.yaml")

    # data
    dm = IMSLightningDataModule(start_date=datetime(*(2023,1,1)), 
                                train_val_split_date=datetime(*(2023,1,2)), 
                                train_test_split_date=datetime(*(2023,1,3)), 
                                end_date=datetime(*(2023,1,9)), 
                                seq_len=20)
    dm.prepare_data()
    dm.setup()

    # train model
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model=l_model, train_dataloaders=dm.train_dataloader())



if __name__ == "__main__":
    main()
