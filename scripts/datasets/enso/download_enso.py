"""
We made a copy of ICAR-ENSO dataset available via AWS S3: https://deep-earth.s3.amazonaws.com/datasets/icar_enso_2021/enso_round1_train_20210201.zip
Alternatively, you may download the dataset directly following the instructions on the official website: https://tianchi.aliyun.com/dataset/dataDetail?dataId=98942
"""
import argparse
from earthformer.datasets.enso.enso_dataloader import download_ENSO


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default=None, type=str)
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    download_ENSO(save_dir=args.save)

if __name__ == "__main__":
    main()
