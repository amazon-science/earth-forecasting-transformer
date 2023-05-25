import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from torch.optim import AdamW
from torch.nn import functional as F

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
        
        # self._max_train_iter = total_num_steps

        # ----- load configs from file, and also from methods bc amazon are dumb  ----- #
        
        # TODO: these lines were replaced with one line for now - 
        # when class is done,  make sure we still want this
        # if oc_file is not None:
        #     oc_from_file = OmegaConf.load(open(oc_file, "r"))
        # else:
        #     oc_from_file = None
        # oc = self.get_base_config(oc_from_file=oc_from_file) 
        # model_cfg = OmegaConf.to_object(oc.model)
        model_cfg = OmegaConf.to_object(self.get_model_config())
        
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

        # ---------------------------- create model object  --------------------------- #

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

        # ------- i HAVENT imported the reset of CuboidSEVIRPLModule's __init__ ------- #
        # --- it consists of another read of the config file & data member creation --- #



    @staticmethod
    def get_dataset_config():

        oc = OmegaConf.create()

        # oc.dataset_name = "ims"
        oc.img_height = 600
        oc.img_width = 600
        oc.in_len = 13
        oc.out_len = 12
        # oc.seq_len = 25
        # oc.plot_stride = 2
        # oc.interval_real_time = 5
        # oc.sample_mode = "sequent"
        # oc.stride = oc.out_len
        # oc.layout = "NTHWC"
        # oc.start_date = None
        # oc.train_val_split_date = (2019, 1, 1)
        # oc.train_test_split_date = (2019, 1, 9)
        # oc.end_date = None
        # oc.metrics_mode = "0"
        # oc.metrics_list = ('csi', 'pod', 'sucr', 'bias')
        # oc.threshold_list = (16, 74, 133, 160, 181, 219)

        return oc
    
    @classmethod
    def get_model_config(cls):

        # TODO: scan all lines and remove unnecessary lines
        cfg = OmegaConf.create()
        dataset_oc = cls.get_dataset_config()
        height = dataset_oc.img_height
        width = dataset_oc.img_width
        in_len = dataset_oc.in_len
        out_len = dataset_oc.out_len
        data_channels = 1
        cfg.input_shape = (in_len, height, width, data_channels)
        cfg.target_shape = (out_len, height, width, data_channels)

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
        cfg.dec_cross_last_n_frames = None

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

        cfg.z_init_method = 'zeros'
        cfg.checkpoint_level = 2
        # initial downsample and final upsample
        cfg.initial_downsample_type = "stack_conv"
        cfg.initial_downsample_activation = "leaky"
        cfg.initial_downsample_stack_conv_num_layers = 3
        cfg.initial_downsample_stack_conv_dim_list = [4, 16, cfg.base_units]
        cfg.initial_downsample_stack_conv_downscale_list = [3, 2, 2]
        cfg.initial_downsample_stack_conv_num_conv_list = [2, 2, 2]
        # initialization
        cfg.attn_linear_init_mode = "0"
        cfg.ffn_linear_init_mode = "0"
        cfg.conv_init_mode = "0"
        cfg.down_up_linear_init_mode = "0"
        cfg.norm_init_mode = "0"
        return cfg


    def forward(self, x):
        return self.torch_nn_module(x)

    def training_step(self, batch, batch_idx):
        x, y = batch # TODO: understand what it means
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
    l_model = CuboidIMSModule()

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
