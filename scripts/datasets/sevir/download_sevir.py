import argparse
from earthformer.datasets.sevir.sevir_torch_wrap import download_SEVIR


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="sevir", type=str)
    parser.add_argument("--save", default=None, type=str)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.dataset == "sevir":
        download_SEVIR(save_dir=args.save)
    else:
        raise ValueError(f"Wrong dataset name {args.dataset}! Must be one of ('sevir', 'sevir_lr').")


if __name__ == "__main__":
    main()
