import os
import argparse
from shutil import copyfile
from omegaconf import OmegaConf
import numpy as np
from earthformer.datasets.nbody.nbody_mnist_torch_wrap import NBodyMovingMNISTLightningDataModule
from earthformer.visualization.utils import save_gif


def duplicate_single_seq_dataset(npz_path, num_copies=64, save_path=None):
    r"""
    Duplicate a single N-body sequence `num_copies` times as a toy dataset for debugging.
    """
    if save_path is None:
        from pathlib import Path
        save_dir = os.path.dirname(npz_path)
        save_name = Path(npz_path).stem + f"_copy{num_copies}.npz"
        save_path = os.path.join(save_dir, save_name)
    f = np.load(npz_path)
    saved_data_dict = dict(f)
    new_data_dict = {}
    for key, val in saved_data_dict.items():
        dup_val = val[0]
        dup_val = np.repeat(dup_val[np.newaxis],
                            repeats=num_copies,
                            axis=0)
        new_data_dict[key] = dup_val
    np.savez_compressed(file=save_path, **new_data_dict)

def generate_nbody_dataset(save_dir=None, oc_file_path=None):
    if oc_file_path is None:
        oc_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cfg.yaml")
    dataset_oc = OmegaConf.to_object(OmegaConf.load(open(oc_file_path, "r")).dataset)
    if save_dir is None:
        from earthformer.config import cfg
        save_dir = os.path.join(cfg.datasets_dir, "nbody")
    data_dir = os.path.join(save_dir, dataset_oc["dataset_name"])
    if os.path.exists(data_dir):
        raise ValueError(f"data_dir {data_dir} already exists!")
    else:
        os.makedirs(data_dir)
    copyfile(oc_file_path, os.path.join(data_dir, "nbody_dataset_cfg.yaml"))
    dm = NBodyMovingMNISTLightningDataModule(
        data_dir=data_dir,
        force_regenerate=False,
        num_train_samples=dataset_oc["num_train_samples"],
        num_val_samples=dataset_oc["num_val_samples"],
        num_test_samples=dataset_oc["num_test_samples"],
        digit_num=dataset_oc["digit_num"],
        img_size=dataset_oc["img_size"],
        raw_img_size=dataset_oc["raw_img_size"],
        seq_len=dataset_oc["seq_len"],
        raw_seq_len_multiplier=dataset_oc["raw_seq_len_multiplier"],
        distractor_num=dataset_oc["distractor_num"],
        distractor_size=dataset_oc["distractor_size"],
        max_velocity_scale=dataset_oc["max_velocity_scale"],
        initial_velocity_range=dataset_oc["initial_velocity_range"],
        random_acceleration_range=dataset_oc["random_acceleration_range"],
        scale_variation_range=dataset_oc["scale_variation_range"],
        rotation_angle_range=dataset_oc["rotation_angle_range"],
        illumination_factor_range=dataset_oc["illumination_factor_range"],
        period=dataset_oc["period"],
        global_rotation_prob=dataset_oc["global_rotation_prob"],
        index_range=dataset_oc["index_range"],
        mnist_data_path=dataset_oc["mnist_data_path"],
        # N-Body params
        nbody_acc_mode=dataset_oc["nbody_acc_mode"],
        nbody_G=dataset_oc["nbody_G"],
        nbody_softening_distance=dataset_oc["nbody_softening_distance"],
        nbody_mass=dataset_oc["nbody_mass"],
        # datamodule_only
        batch_size=1,
        num_workers=8, )
    dm.prepare_data()
    dm.setup()
    seq = dm.nbody_train[0]
    save_gif(single_seq=seq[:, :, :, 0].numpy().astype(np.float32),
             fname=os.path.join(data_dir, "nbody_example.gif"))

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None, type=str)
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    generate_nbody_dataset(oc_file_path=args.cfg)

if __name__ == "__main__":
    main()
