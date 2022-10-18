from typing import Optional, Union, Tuple
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
import torch
from torch.utils.data import Dataset, DataLoader
from subseasonal_data import data_loaders
from subseasonal_data.utils import get_measurement_variable
from subseasonal_toolkit.utils.general_util import get_task_from_string


def get_data_matrix(data, values):
    """Get pandas dataframe with (lat, lon, values) ready for plotting

    If there is more than one value per (lat, lon) grid point, the values will be averaged.
    This is especially useful for calculating daily/monthly/yearly averages.

    Parameters
    ----------
    data:   pd.DataFrame
        with (lat, lon, values) format

    values: str
        Name of the 'values' column

    Returns
    data_matrix:    np.ma.core.MaskedArray
        Shape = (lat, lon)
    """
    # Average if more than one data point per (lat, lon) pair
    data_aux = data[["lat", "lon", values]].groupby(by=["lat", "lon"], as_index=False).agg(np.mean)
    data_pivot = data_aux.pivot(index='lat', columns='lon', values=values)
    data_matrix = data_pivot.values
    data_matrix = np.ma.masked_invalid(data_matrix)
    return data_matrix

class SubseasonalTorchDataset(Dataset):

    def __init__(self,
                 task: str = "us_precip_34w",
                 interval: Union[DateOffset, Tuple] = DateOffset(months=1),
                 start_date: Optional[pd.Timestamp] = None,
                 end_date: Optional[pd.Timestamp] = None):
        r"""
        Parameters
        ----------
        task:    str
            f"{region}_{modality}_{horizon}"
        start_date: pd.Timestamp
            Data entries with datetime >= start_date are included.
        end_date:   pd.Timestamp
            Data entries with datetime < end_date are included.
        """
        super(SubseasonalTorchDataset, self).__init__()
        region, modality, horizon = get_task_from_string(task)
        gt_id = f"{region}_{modality}"
        # get pd.DataFrame
        self.start_date = start_date
        self.end_date = end_date
        var_name = get_measurement_variable(gt_id)
        df = data_loaders.get_ground_truth(gt_id).loc[
                :,
                ["start_date", "lat", "lon", var_name]]
        if start_date is not None:
            df = df[df.start_date >= start_date]
        if end_date is not None:
            df = df[df.start_date < end_date]
        self.gt_df = df
        # Set time interval between adjacent frames
        if isinstance(interval, tuple):
            # Parameters that **add** to the offset (like Timedelta):
            date_offset_keys = ("years", "months", "weeks", "days", "hours", "minutes", "seconds", "microseconds", "nanoseconds")
            interval = DateOffset(**dict(zip(date_offset_keys, interval)))
        self.interval = interval

    def __getitem__(self, item):

        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

