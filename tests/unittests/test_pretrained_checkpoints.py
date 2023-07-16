import warnings
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
from earthformer.utils.utils import download
from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel
from earthformer.cuboid_transformer.cuboid_transformer_unet_dec import CuboidTransformerAuxModel


NUM_TEST_ITER = 16  # max = 32 since saved `unittest_data.pt` only contains the first 0 to 31 data entries.
test_data_dir = os.path.join(cfg.root_dir, "tests", "unittests", "test_pretrained_checkpoints_data")

def s3_download_unittest_data(data_name):
    test_data_path = os.path.join(test_data_dir, data_name)
    if not os.path.exists(test_data_path):
        os.makedirs(test_data_dir, exist_ok=True)
        download(url=f"s3://earthformer/unittests/{data_name}", path=test_data_path)


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
    pretrained_ckpt_name = "earthformer_sevir.pt"
    test_data_name = "unittest_sevir_data_bs1_idx0to31.pt"
    s3_download_unittest_data(data_name=test_data_name)
    test_data_path = os.path.join(test_data_dir, test_data_name)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # Load pretrained model
    pretrained_cfg_path = os.path.join(cfg.root_dir, "scripts", "cuboid_transformer", "sevir", "earthformer_sevir_v1.yaml")
    pretrained_cfg = OmegaConf.load(open(pretrained_cfg_path, "r"))
    model = config_cuboid_transformer(
        cfg=pretrained_cfg,
        model_type="CuboidTransformerModel").to(device)
    model.eval()
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
    layout_cfg = pretrained_cfg.layout
    in_slice, out_slice = layout_to_in_out_slice(layout=layout_cfg.layout,
                                                 in_len=layout_cfg.in_len,
                                                 out_len=layout_cfg.out_len)
    test_mse_metrics = torchmetrics.MeanSquaredError().to(device)
    test_mae_metrics = torchmetrics.MeanAbsoluteError().to(device)
    test_data = torch.load(test_data_path)
    counter = 0
    with torch.no_grad():
        for batch in test_data:
            data_seq = batch['vil'].contiguous().to(device)
            x = data_seq[in_slice]
            y = data_seq[out_slice]
            y_hat = model(x)
            test_mse_metrics(y_hat, y)
            test_mae_metrics(y_hat, y)
            counter += 1
            if counter >= NUM_TEST_ITER:
                break
    test_mse = test_mse_metrics.compute()
    test_mae = test_mae_metrics.compute()
    assert test_mse < 1E-2
    assert test_mae < 5E-2

def test_enso():
    pretrained_ckpt_name = "earthformer_icarenso2021.pt"
    test_data_name = "unittest_icarenso2021_data_bs1_idx0to31.pt"
    s3_download_unittest_data(data_name=test_data_name)
    test_data_path = os.path.join(test_data_dir, test_data_name)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # Load pretrained model
    pretrained_cfg_path = os.path.join(cfg.root_dir, "scripts", "cuboid_transformer", "enso", "earthformer_enso_v1.yaml")
    pretrained_cfg = OmegaConf.load(open(pretrained_cfg_path, "r"))
    model = config_cuboid_transformer(
        cfg=pretrained_cfg,
        model_type="CuboidTransformerModel").to(device)
    model.eval()

    if not os.path.exists(os.path.join(cfg.pretrained_checkpoints_dir, pretrained_ckpt_name)):
        s3_download_pretrained_ckpt(ckpt_name=pretrained_ckpt_name,
                                    save_dir=cfg.pretrained_checkpoints_dir,
                                    exist_ok=False)
    state_dict = torch.load(os.path.join(cfg.pretrained_checkpoints_dir, pretrained_ckpt_name),
                            map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict=state_dict, strict=False)
    assert len(missing_keys) == 0, f"missing_keys {missing_keys} when loading pretrained state_dict."
    assert len(unexpected_keys) == 0, f"missing_keys {unexpected_keys} when loading pretrained state_dict."
    # Test on ENSO test
    layout_cfg = pretrained_cfg.layout
    in_slice, out_slice = layout_to_in_out_slice(layout=layout_cfg.layout,
                                                 in_len=layout_cfg.in_len,
                                                 out_len=layout_cfg.out_len)
    test_mse_metrics = torchmetrics.MeanSquaredError().to(device)
    test_mae_metrics = torchmetrics.MeanAbsoluteError().to(device)
    test_data = torch.load(test_data_path)
    counter = 0
    with torch.no_grad():
        for batch in test_data:
            sst_seq, nino_target = batch
            data_seq = sst_seq.float().unsqueeze(-1).to(device)
            x = data_seq[in_slice]
            y = data_seq[out_slice]
            y_hat = model(x)
            test_mse_metrics(y_hat, y)
            test_mae_metrics(y_hat, y)
            counter += 1
            if counter >= NUM_TEST_ITER:
                break
    test_mse = test_mse_metrics.compute()
    test_mae = test_mae_metrics.compute()
    assert test_mse < 5E-4
    assert test_mae < 2E-2

def test_earthnet():
    data_channels = 4
    pretrained_ckpt_name = "earthformer_earthnet2021.pt"
    test_data_name = "unittest_earthnet2021_data_bs1_idx0to31.pt"
    s3_download_unittest_data(data_name=test_data_name)
    test_data_path = os.path.join(test_data_dir, test_data_name)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # Load pretrained model
    pretrained_cfg_path = os.path.join(cfg.root_dir, "scripts", "cuboid_transformer", "earthnet_w_meso", "earthformer_earthnet_v1.yaml")
    pretrained_cfg = OmegaConf.load(open(pretrained_cfg_path, "r"))
    model = config_cuboid_transformer(
        cfg=pretrained_cfg,
        model_type="CuboidTransformerAuxModel").to(device)
    model.eval()

    if not os.path.exists(os.path.join(cfg.pretrained_checkpoints_dir, pretrained_ckpt_name)):
        s3_download_pretrained_ckpt(ckpt_name=pretrained_ckpt_name,
                                    save_dir=cfg.pretrained_checkpoints_dir,
                                    exist_ok=False)
    state_dict = torch.load(os.path.join(cfg.pretrained_checkpoints_dir, pretrained_ckpt_name),
                            map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict=state_dict, strict=False)
    assert len(missing_keys) == 0, f"missing_keys {missing_keys} when loading pretrained state_dict."
    assert len(unexpected_keys) == 0, f"missing_keys {unexpected_keys} when loading pretrained state_dict."
    # Test on EarthNet2021 test
    layout_cfg = pretrained_cfg.layout
    in_slice, out_slice = layout_to_in_out_slice(layout=layout_cfg.layout,
                                                 in_len=layout_cfg.in_len,
                                                 out_len=layout_cfg.out_len)
    test_mse_metrics = torchmetrics.MeanSquaredError().to(device)
    test_mae_metrics = torchmetrics.MeanAbsoluteError().to(device)
    test_data = torch.load(test_data_path)
    counter = 0
    with torch.no_grad():
        for batch in test_data:
            highresdynamic = batch["highresdynamic"].to(device)
            seq = highresdynamic[..., :data_channels]
            print(seq.shape)
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
            counter += 1
            if counter >= NUM_TEST_ITER:
                break
    test_mse = test_mse_metrics.compute()
    test_mae = test_mae_metrics.compute()
    assert test_mse < 5E-4
    assert test_mae < 1E-2
