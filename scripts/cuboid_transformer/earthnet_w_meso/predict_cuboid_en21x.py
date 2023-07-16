

import xarray as xr

import torch
import os
from train_cuboid_earthnet import CuboidEarthNet2021PLModule
from earthformer.utils.utils import download
from omegaconf import OmegaConf
from earthformer.datasets.earthnet.earthnet21x_dataloader import EarthNet2021xLightningDataModule, EarthNet2021xTestDataset
from pytorch_lightning import Trainer, seed_everything, loggers as pl_loggers
from earthformer.datasets.earthnet.visualization import vis_earthnet_seq
import numpy as np
from pathlib import Path
from tqdm import tqdm


def process_sample(sample):
    # High-resolution Earth surface data. The channels are [blue, green, red, nir, cloud]
    highresdynamic = sample['highresdynamic']
    highresstatic = sample['highresstatic']

    # The meso-scale data. The channels are ["precipitation", "pressure", "temp mean", "temp min", "temp max"]
    mesodynamic = sample['mesodynamic']
    mesostatic = sample['mesostatic']

    highresdynamic = np.nan_to_num(highresdynamic, nan=0.0, posinf=1.0, neginf=0.0)
    highresdynamic = np.clip(highresdynamic, a_min=0.0, a_max=1.0)
    mesodynamic = np.nan_to_num(mesodynamic, nan=0.0)
    highresstatic = np.nan_to_num(highresstatic, nan=0.0)
    mesostatic = np.nan_to_num(mesostatic, nan=0.0)
    return highresdynamic, highresstatic, mesodynamic, mesostatic

def main():
    print("Loading Config")
    pred_dir = Path("./experiments/preds_en21x/")

    config_file = "./earthformer_earthnet_v1.yaml"
    config = OmegaConf.load(open(config_file, "r"))
    in_len = config.layout.in_len
    out_len = config.layout.out_len

    seed = config.optim.seed
    dataset_cfg = OmegaConf.to_object(config.dataset)

    seed_everything(seed, workers=True)

    micro_batch_size = 1
    print("Loading Dataset")
    earthnet_iid_testset = EarthNet2021xTestDataset(subset_name="iid",
                                                data_dir="/Net/Groups/BGI/work_1/scratch/s3/earthnet/earthnet2021x/iid/",
                                                layout=config.dataset.layout,
                                                static_layout=config.dataset.static_layout,
                                                highresstatic_expand_t=config.dataset.highresstatic_expand_t,
                                                mesostatic_expand_t=config.dataset.mesostatic_expand_t,
                                                meso_crop=None,
                                                fp16=False)

    save_dir = "./experiments"
    print("Loading Model")
    pl_module = CuboidEarthNet2021PLModule(
        total_num_steps=None,
        save_dir="./experiments",
        oc_file=config_file
    )

    pretrained_checkpoint_url = "https://earthformer.s3.amazonaws.com/pretrained_checkpoints/earthformer_earthnet2021.pt"
    local_checkpoint_path = os.path.join(save_dir, "earthformer_earthnet2021.pt")
    download(url=pretrained_checkpoint_url, path=local_checkpoint_path)

    state_dict = torch.load(local_checkpoint_path, map_location=torch.device("cpu"))
    pl_module.torch_nn_module.load_state_dict(state_dict=state_dict)

    pl_module.torch_nn_module.cuda()
    pl_module.torch_nn_module.eval()
    print("Starting Predictions")
    for idx in tqdm(range(len(earthnet_iid_testset))):
        highresdynamic, highresstatic, mesodynamic, mesostatic = process_sample(earthnet_iid_testset[idx])


        with torch.no_grad():
            pred_seq, loss, in_seq, target_seq, mask = pl_module({"highresdynamic": torch.tensor(np.expand_dims(highresdynamic, axis=0)).cuda(), 
                                                                "highresstatic": torch.tensor(np.expand_dims(highresstatic, axis=0)).cuda(),
                                                                "mesodynamic": torch.tensor(np.expand_dims(mesodynamic, axis=0)).cuda(),
                                                                "mesostatic": torch.tensor(np.expand_dims(mesostatic, axis=0)).cuda()})
            pred_seq_np = pred_seq.detach().cpu().numpy()

        targ_path = Path(earthnet_iid_testset.nc_path_list[idx])

        targ_cube = xr.open_dataset(targ_path)

        lat = targ_cube.lat
        lon = targ_cube.lon

        blue_pred = pred_seq_np[0,:,:,:,0]
        green_pred = pred_seq_np[0,:,:,:,1]
        red_pred = pred_seq_np[0,:,:,:,2]
        nir_pred = pred_seq_np[0,:,:,:,3]
        ndvi_pred = ((nir_pred - red_pred) / (nir_pred + red_pred + 1e-8))
        
        pred_cube = xr.Dataset({"ndvi_pred": xr.DataArray(data = ndvi_pred, coords = {"time": targ_cube.time.isel(time = slice(4,None,5)).isel(time = slice(in_len, in_len + out_len)), "lat": lat, "lon": lon}, dims = ["time","lat", "lon"])})

        
        pred_path = pred_dir/targ_path.parent.stem/targ_path.name
        pred_path.parent.mkdir(parents = True, exist_ok = True)
        if not pred_path.is_file():
            pred_cube.to_netcdf(pred_path, encoding={"ndvi_pred":{"dtype": "float32"}})

if __name__ == "__main__":
    main()