import os
import unittest
from omegaconf import OmegaConf
import numpy as np
import torch
import torchmetrics
from earthformer.config import cfg
from earthformer.utils.checkpoint import s3_download_pretrained_ckpt
from earthformer.utils.layout import layout_to_in_out_slice
from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel
from earthformer.cuboid_transformer.cuboid_transformer_unet_dec import CuboidTransformerAuxModel
from earthformer.metrics.sevir import SEVIRSkillScore
from earthformer.datasets.sevir.sevir_torch_wrap import SEVIRLightningDataModule


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

class CuboidTransformerBackwardCompatibility(unittest.TestCase):

    def test_sevir(self):
        micro_batch_size = 8
        pretrained_ckpt_name = "earthformer_sevir.pt"
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        # Load pretrained model
        pretrained_cfg_path = os.path.join(cfg.root_dir, "scripts", "cuboid_transformer", "sevir", "cfg_pretrained.yaml")
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
        self.assertEqual(len(missing_keys), 0)
        self.assertEqual(len(unexpected_keys), 0)
        # Test on SEVIR test
        dataset_cfg = OmegaConf.to_object(pretrained_cfg.dataset)
        layout_cfg = OmegaConf.to_object(pretrained_cfg.layout)
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
        in_slice, out_slice = layout_to_in_out_slice(layout=layout_cfg["layout"],
                                                     in_len=layout_cfg["in_len"],
                                                     out_len=layout_cfg["out_len"])
        threshold_list = dataset_cfg["threshold_list"],
        metrics_list = dataset_cfg["metrics_list"],
        test_mse_metrics = torchmetrics.MeanSquaredError().to(device)
        test_mae_metrics = torchmetrics.MeanAbsoluteError().to(device)
        test_score_metrics = SEVIRSkillScore(
            mode=dataset_cfg["metrics_mode"],
            seq_len=dataset_cfg["out_len"],
            layout=dataset_cfg["layout"],
            threshold_list=threshold_list,
            metrics_list=metrics_list,
            eps=1e-4, ).to(device)
        for batch in dm.test_dataloader():
            data_seq = batch['vil'].contiguous().to(device)
            x = data_seq[in_slice]
            y = data_seq[out_slice]
            y_hat = model(x)
            test_mse_metrics(y_hat, y)
            test_mae_metrics(y_hat, y)
            test_score_metrics(y_hat, y)
        test_mse = test_mse_metrics.compute()
        test_mae = test_mae_metrics.compute()
        print(f"test_mse = {test_mse}")
        print(f"test_mae = {test_mae}")
        test_score = test_score_metrics.compute()
        for metrics in metrics_list:
            for thresh in threshold_list:
                score_mean = np.mean(test_score[thresh][metrics]).item()
                print(f"test_{metrics}_{thresh} = {score_mean}")
            score_avg_mean = test_score.get("avg", None)
            if score_avg_mean is not None:
                score_avg_mean = np.mean(score_avg_mean[metrics]).item()
                print(f"test_{metrics}_avg = {score_avg_mean}")

if __name__ == '__main__':
    unittest.main()
