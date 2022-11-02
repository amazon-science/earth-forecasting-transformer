import os
from omegaconf import OmegaConf
import pytest
import numpy as np
import torch
from torch.nn import functional as F
import torchmetrics
from einops import rearrange
from earthformer.config import cfg
from earthformer.utils.checkpoint import s3_download_pretrained_ckpt
from earthformer.utils.layout import layout_to_in_out_slice
from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel
from earthformer.cuboid_transformer.cuboid_transformer_unet_dec import CuboidTransformerAuxModel
from earthformer.datasets.sevir.sevir_torch_wrap import SEVIRLightningDataModule
from earthformer.datasets.enso.enso_dataloader import ENSOLightningDataModule
from earthformer.datasets.earthnet.earthnet_dataloader import get_EarthNet2021_dataloaders


def config_cuboid_transformer(cfg, model_type="CuboidTransformerModel"):
    model_cfg = OmegaConf.to_object(cfg.model)
    num_blocks = len(model_cfg["enc_depth"])
    if isinstance(model_cfg["self_pattern"], str):
        enc_attn_patterns = [model_cfg.pop("self_pattern")] * num_blocks
    else:
        enc_attn_patterns = OmegaConf.to_container(model_cfg.pop("self_pattern"))
    model_cfg["enc_attn_patterns"] = enc_attn_patterns
    if isinstance(model_cfg["cross_self_pattern"], str):
        dec_self_attn_patterns = [model_cfg.pop("cross_self_pattern")] * num_blocks
    else:
        dec_self_attn_patterns = OmegaConf.to_container(model_cfg.pop("cross_self_pattern"))
    model_cfg["dec_self_attn_patterns"] = dec_self_attn_patterns
    if isinstance(model_cfg["cross_pattern"], str):
        dec_cross_attn_patterns = [model_cfg.pop("cross_pattern")] * num_blocks
    else:
        dec_cross_attn_patterns = OmegaConf.to_container(model_cfg.pop("cross_pattern"))
    model_cfg["dec_cross_attn_patterns"] = dec_cross_attn_patterns
    if model_type == "CuboidTransformerModel":
        model = CuboidTransformerModel(**model_cfg)
    elif model_type == "CuboidTransformerAuxModel":
        model = CuboidTransformerAuxModel(**model_cfg)
    else:
        raise ValueError(f"Invalid model_type {model_type}. Must be 'CuboidTransformerModel' or ''.")
    return model

def test_sevir():
    micro_batch_size = 1
    pretrained_ckpt_name = "earthformer_sevir.pt"
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # Load pretrained model
    pretrained_cfg_path = os.path.join(cfg.root_dir, "scripts", "cuboid_transformer", "sevir", "earthformer_sevir_v1.yaml")
    pretrained_cfg = OmegaConf.load(open(pretrained_cfg_path, "r"))
    model = config_cuboid_transformer(
        cfg=pretrained_cfg,
        model_type="CuboidTransformerModel").to(device)

    if not os.path.exists(os.path.join(cfg.pretrained_checkpoints_dir, pretrained_ckpt_name)):
        s3_download_pretrained_ckpt(ckpt_name=pretrained_ckpt_name,
                                    save_dir=cfg.pretrained_checkpoints_dir,
                                    exist_ok=False)
    state_dict = torch.load(os.path.join(cfg.pretrained_checkpoints_dir, pretrained_ckpt_name),
                            map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict=state_dict, strict=False)
    assert len(missing_keys) == 0, f"missing_keys {missing_keys} when loading pretrained state_dict."
    assert len(unexpected_keys) == 0, f"missing_keys {unexpected_keys} when loading pretrained state_dict."
    # Test on SEVIR test
    dataset_cfg = OmegaConf.to_object(pretrained_cfg.dataset)
    layout_cfg = pretrained_cfg.layout
    dm = SEVIRLightningDataModule(
        seq_len=dataset_cfg["seq_len"],
        sample_mode=dataset_cfg["sample_mode"],
        stride=dataset_cfg["stride"],
        batch_size=micro_batch_size,
        layout=dataset_cfg["layout"],
        output_type=np.float32,
        preprocess=True,
        rescale_method="01",
        verbose=False,
        # datamodule_only
        dataset_name=dataset_cfg["dataset_name"],
        start_date=dataset_cfg["start_date"],
        train_val_split_date=dataset_cfg["train_val_split_date"],
        train_test_split_date=dataset_cfg["train_test_split_date"],
        end_date=dataset_cfg["end_date"],
        num_workers=8, )
    dm.prepare_data()
    dm.setup()
    in_slice, out_slice = layout_to_in_out_slice(layout=layout_cfg.layout,
                                                 in_len=layout_cfg.in_len,
                                                 out_len=layout_cfg.out_len)
    test_mse_metrics = torchmetrics.MeanSquaredError().to(device)
    test_mae_metrics = torchmetrics.MeanAbsoluteError().to(device)

    for batch in dm.test_dataloader():
        data_seq = batch['vil'].contiguous().to(device)
        x = data_seq[in_slice]
        y = data_seq[out_slice]
        y_hat = model(x)
        test_mse_metrics(y_hat, y)
        test_mae_metrics(y_hat, y)
        break

    test_mse = test_mse_metrics.compute()
    test_mae = test_mae_metrics.compute()
    assert test_mse < 3E-5
    assert test_mae < 3E-3

def test_enso():
    micro_batch_size = 1
    pretrained_ckpt_name = "earthformer_icarenso2021.pt"
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # Load pretrained model
    pretrained_cfg_path = os.path.join(cfg.root_dir, "scripts", "cuboid_transformer", "enso", "earthformer_enso_v1.yaml")
    pretrained_cfg = OmegaConf.load(open(pretrained_cfg_path, "r"))
    model = config_cuboid_transformer(
        cfg=pretrained_cfg,
        model_type="CuboidTransformerModel").to(device)

    if not os.path.exists(os.path.join(cfg.pretrained_checkpoints_dir, pretrained_ckpt_name)):
        s3_download_pretrained_ckpt(ckpt_name=pretrained_ckpt_name,
                                    save_dir=cfg.pretrained_checkpoints_dir,
                                    exist_ok=False)
    state_dict = torch.load(os.path.join(cfg.pretrained_checkpoints_dir, pretrained_ckpt_name),
                            map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict=state_dict, strict=False)
    assert len(missing_keys) == 0, f"missing_keys {missing_keys} when loading pretrained state_dict."
    assert len(unexpected_keys) == 0, f"missing_keys {unexpected_keys} when loading pretrained state_dict."
    # Test on SEVIR test
    dataset_cfg = OmegaConf.to_object(pretrained_cfg.dataset)
    layout_cfg = pretrained_cfg.layout
    dm = ENSOLightningDataModule(
        in_len=dataset_cfg["in_len"],
        out_len=dataset_cfg["out_len"],
        in_stride=dataset_cfg["in_stride"],
        out_stride=dataset_cfg["out_stride"],
        train_samples_gap=dataset_cfg["train_samples_gap"],
        eval_samples_gap=dataset_cfg["eval_samples_gap"],
        normalize_sst=dataset_cfg["normalize_sst"],
        batch_size=micro_batch_size,
        num_workers=1)
    dm.prepare_data()
    dm.setup()
    in_slice, out_slice = layout_to_in_out_slice(layout=layout_cfg.layout,
                                                 in_len=layout_cfg.in_len,
                                                 out_len=layout_cfg.out_len)
    test_mse_metrics = torchmetrics.MeanSquaredError().to(device)
    test_mae_metrics = torchmetrics.MeanAbsoluteError().to(device)

    for batch in dm.test_dataloader():
        sst_seq, nino_target = batch
        data_seq = sst_seq.float().unsqueeze(-1).to(device)
        x = data_seq[in_slice]
        y = data_seq[out_slice]
        y_hat = model(x)
        test_mse_metrics(y_hat, y)
        test_mae_metrics(y_hat, y)
        break

    test_mse = test_mse_metrics.compute()
    test_mae = test_mae_metrics.compute()
    assert test_mse < 5E-4
    assert test_mae < 2E-2

def test_earthnet():
    micro_batch_size = 1
    data_channels = 4
    pretrained_ckpt_name = "earthformer_earthnet2021.pt"
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # Load pretrained model
    pretrained_cfg_path = os.path.join(cfg.root_dir, "scripts", "cuboid_transformer", "earthnet_w_meso", "earthformer_earthnet_v1.yaml")
    pretrained_cfg = OmegaConf.load(open(pretrained_cfg_path, "r"))
    model = config_cuboid_transformer(
        cfg=pretrained_cfg,
        model_type="CuboidTransformerAuxModel").to(device)

    if not os.path.exists(os.path.join(cfg.pretrained_checkpoints_dir, pretrained_ckpt_name)):
        s3_download_pretrained_ckpt(ckpt_name=pretrained_ckpt_name,
                                    save_dir=cfg.pretrained_checkpoints_dir,
                                    exist_ok=False)
    state_dict = torch.load(os.path.join(cfg.pretrained_checkpoints_dir, pretrained_ckpt_name),
                            map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict=state_dict, strict=False)
    assert len(missing_keys) == 0, f"missing_keys {missing_keys} when loading pretrained state_dict."
    assert len(unexpected_keys) == 0, f"missing_keys {unexpected_keys} when loading pretrained state_dict."
    # Test on SEVIR test
    dataset_cfg = OmegaConf.to_object(pretrained_cfg.dataset)
    layout_cfg = pretrained_cfg.layout
    test_dataloader = get_EarthNet2021_dataloaders(
        dataloader_return_mode=dataset_cfg["return_mode"],
        data_aug_mode=dataset_cfg["data_aug_mode"],
        data_aug_cfg=dataset_cfg["data_aug_cfg"],
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
        num_workers=8,
    )["test_dataloader"][0]
    in_slice, out_slice = layout_to_in_out_slice(layout=layout_cfg.layout,
                                                 in_len=layout_cfg.in_len,
                                                 out_len=layout_cfg.out_len)
    test_mse_metrics = torchmetrics.MeanSquaredError().to(device)
    test_mae_metrics = torchmetrics.MeanAbsoluteError().to(device)

    for batch in test_dataloader:
        highresdynamic = batch["highresdynamic"].to(device)
        seq = highresdynamic[..., :data_channels]
        # mask from dataloader: 1 for mask and 0 for non-masked
        mask = highresdynamic[..., data_channels: data_channels + 1][out_slice]

        in_seq = seq[in_slice]
        target_seq = seq[out_slice]

        # process aux data
        highresstatic = batch["highresstatic"].to(device)  # (b c h w)
        mesodynamic = batch["mesodynamic"].to(device)  # (b t h w c)
        mesostatic = batch["mesostatic"].to(device)  # (b c h w)

        mesodynamic_interp = rearrange(mesodynamic,
                                       "b t h w c -> b c t h w")
        mesodynamic_interp = F.interpolate(mesodynamic_interp,
                                           size=(layout_cfg.in_len + layout_cfg.out_len,
                                                 layout_cfg.img_height,
                                                 layout_cfg.img_width),
                                           mode="nearest")
        highresstatic_interp = rearrange(highresstatic,
                                         "b c h w -> b c 1 h w")
        highresstatic_interp = F.interpolate(highresstatic_interp,
                                             size=(layout_cfg.in_len + layout_cfg.out_len,
                                                   layout_cfg.img_height,
                                                   layout_cfg.img_width),
                                             mode="nearest")
        mesostatic_interp = rearrange(mesostatic,
                                      "b c h w -> b c 1 h w")
        mesostatic_interp = F.interpolate(mesostatic_interp,
                                          size=(layout_cfg.in_len + layout_cfg.out_len,
                                                layout_cfg.img_height,
                                                layout_cfg.img_width),
                                          mode="nearest")
        aux_data = torch.cat((highresstatic_interp, mesodynamic_interp, mesostatic_interp),
                             dim=1)
        aux_data = rearrange(aux_data,
                             "b c t h w -> b t h w c")

        pred_seq = model(in_seq, aux_data[in_slice], aux_data[out_slice])
        test_mse_metrics(pred_seq * (1 - mask), target_seq * (1 - mask))
        test_mae_metrics(pred_seq * (1 - mask), target_seq * (1 - mask))
        break

    test_mse = test_mse_metrics.compute()
    test_mae = test_mae_metrics.compute()
    assert test_mse < 5E-4
    assert test_mae < 2E-2
