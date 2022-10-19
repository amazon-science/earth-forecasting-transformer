import datetime

import numpy as np
from matplotlib import pyplot as plt

from earthformer.datasets.sevir.sevir_dataloader import SEVIR_CATALOG, SEVIR_DATA_DIR, SEVIR_RAW_SEQ_LEN, \
    SEVIR_LR_CATALOG, SEVIR_LR_DATA_DIR, SEVIR_LR_RAW_SEQ_LEN, SEVIRDataLoader


def report_SEVIR_statistics(
        dataset="sevir",
        sanity_check=True,
        hist_save_path="tmp_sevir_hist.png"):
    r"""
    Report important statistics of SEVIR dataset, including:
    - The distribution of pixel values (from 0 to 255).

    Refer to https://discuss.pytorch.org/t/plot-a-histogram-for-multiple-images-full-dataset/67600
    """
    if sanity_check:
        start_date = datetime.datetime(2019, 5, 27)
        train_val_split_date = datetime.datetime(2019, 5, 29)
        train_test_split_date = datetime.datetime(2019, 6, 1)
        end_date = datetime.datetime(2019, 6, 3)
    else:
        # total number of event = 14926 for the whole training set
        # val percentage is 20% in original SEVIR paper
        # this split results in 11906 / 14926 = 79.77%
        # if use (2019, 2, 1), the ratio becomes 12489 / 14926 = 83.67%
        start_date = None
        train_val_split_date = datetime.datetime(2019, 1, 1)
        # train_val_split_date = datetime.datetime(2019, 2, 1)
        train_test_split_date = datetime.datetime(2019, 6, 1)
        end_date = None
    if dataset == "sevir":
        catalog_path = SEVIR_CATALOG
        data_dir = SEVIR_DATA_DIR
        raw_seq_len = SEVIR_RAW_SEQ_LEN
    elif dataset == "sevir_lr":
        catalog_path = SEVIR_LR_CATALOG
        data_dir = SEVIR_LR_DATA_DIR
        raw_seq_len = SEVIR_LR_RAW_SEQ_LEN
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    batch_size = 1
    data_types = ["vil", ]
    layout = "NHWT"
    seq_len = raw_seq_len
    stride = seq_len
    sample_mode = "sequent"

    train_dataloader = SEVIRDataLoader(
        sevir_catalog=catalog_path, sevir_data_dir=data_dir,
        raw_seq_len=raw_seq_len, split_mode="uneven",
        data_types=data_types, seq_len=seq_len, stride=stride,
        sample_mode=sample_mode, batch_size=batch_size, layout=layout,
        num_shard=1, rank=0,
        start_date=start_date, end_date=train_val_split_date)
    # val_dataloader = SEVIRDataLoader(
    #     sevir_catalog=catalog_path, sevir_data_dir=data_dir,
    #     raw_seq_len=raw_seq_len, split_mode="uneven",
    #     data_types=data_types, seq_len=seq_len, stride=stride,
    #     sample_mode=sample_mode, batch_size=batch_size, layout=layout,
    #     num_shard=1, rank=0,
    #     start_date=train_val_split_date, end_date=train_test_split_date)
    # test_dataloader = SEVIRDataLoader(
    #     sevir_catalog=catalog_path, sevir_data_dir=data_dir,
    #     raw_seq_len=raw_seq_len, split_mode="uneven",
    #     data_types=data_types, seq_len=seq_len, stride=stride,
    #     sample_mode=sample_mode, batch_size=batch_size, layout=layout,
    #     num_shard=1, rank=0,
    #     start_date=train_test_split_date, end_date=end_date)

    data_loader = train_dataloader
    # data_loader = val_dataloader
    # data_loader = test_dataloader

    num_bins = 256
    count = np.zeros(num_bins)
    for data_idx, data in enumerate(data_loader):
        data_seq = data['vil']
        hist = np.histogram(data_seq, bins=num_bins, range=[0, 255])
        count += hist[0]
    bins = hist[1]
    fig = plt.figure()
    plt.bar(bins[:-1], count, color='b', alpha=0.5)
    plt.savefig(hist_save_path)