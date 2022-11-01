import os
import unittest
import OmegaConf
import torch
from earthformer.config import cfg
from earthformer.utils.checkpoint import s3_download_pretrained_ckpt
from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel


class CuboidTransformerBackwardCompatibility(unittest.TestCase):

    def test_sevir(self):
        pretrained_ckpt_name = "earthformer_sevir.pt"
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        pretrained_cfg_path = os.path.join(cfg.root_dir, "scripts", "cuboid_transformer", "sevir", "cfg_pretrained.yaml")
        model_cfg = OmegaConf.to_object(OmegaConf.load(open(pretrained_cfg_path, "r")).model)
        num_blocks = len(model_cfg["enc_depth"])
        if isinstance(model_cfg["self_pattern"], str):
            enc_attn_patterns = [model_cfg["self_pattern"]] * num_blocks
        else:
            enc_attn_patterns = OmegaConf.to_container(model_cfg["self_pattern"])
        model_cfg["enc_attn_patterns"] = enc_attn_patterns
        if isinstance(model_cfg["cross_self_pattern"], str):
            dec_self_attn_patterns = [model_cfg["cross_self_pattern"]] * num_blocks
        else:
            dec_self_attn_patterns = OmegaConf.to_container(model_cfg["cross_self_pattern"])
        model_cfg["dec_self_attn_patterns"] = dec_self_attn_patterns
        if isinstance(model_cfg["cross_pattern"], str):
            dec_cross_attn_patterns = [model_cfg["cross_pattern"]] * num_blocks
        else:
            dec_cross_attn_patterns = OmegaConf.to_container(model_cfg["cross_pattern"])
        model_cfg["dec_cross_attn_patterns"] = dec_cross_attn_patterns
        model = CuboidTransformerModel(**model_cfg).to(device)

        if not os.path.exists(os.path.join(cfg.pretrained_checkpoints_dir, pretrained_ckpt_name)):
            s3_download_pretrained_ckpt(ckpt_name=pretrained_ckpt_name,
                                        save_dir=cfg.pretrained_checkpoints_dir,
                                        exist_ok=False)
        state_dict = torch.load(os.path.join(cfg.pretrained_checkpoints_dir, pretrained_ckpt_name),
                                map_location=device)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict=state_dict)
        self.assertEqual(len(missing_keys), 0)
        self.assertEqual(len(unexpected_keys), 0)

if __name__ == '__main__':
    unittest.main()
