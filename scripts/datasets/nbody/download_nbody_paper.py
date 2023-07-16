import warnings
import os


nbody_paper_name = "nbody_digits3_len20_size64_r0_train20k"
nbody_paper_zip_name = "nbody_paper.zip"

def s3_download_nbody_paper(save_dir=None, exist_ok=False):
    if save_dir is None:
        from earthformer.config import cfg
        save_dir = os.path.join(cfg.datasets_dir, "nbody")
    if os.path.exists(os.path.join(save_dir, nbody_paper_name)) and not exist_ok:
        warnings.warn(f"N-body dataset {os.path.join(save_dir, nbody_paper_name)} already exists!")
    else:
        os.makedirs(save_dir, exist_ok=True)
        os.system(f"aws s3 cp --no-sign-request s3://earthformer/nbody/{nbody_paper_zip_name} "
                  f"{save_dir}")
        os.system(f"unzip {os.path.join(save_dir, nbody_paper_zip_name)} "
                  f"-d {save_dir}")

if __name__ == "__main__":
    s3_download_nbody_paper(exist_ok=False)
