from copy import deepcopy
import numpy as np
import torch


def layout_to_in_out_slice(layout, in_len, out_len=None):
    t_axis = layout.find("T")
    num_axes = len(layout)
    in_slice = [slice(None, None), ] * num_axes
    out_slice = deepcopy(in_slice)
    in_slice[t_axis] = slice(None, in_len)
    if out_len is None:
        out_slice[t_axis] = slice(in_len, None)
    else:
        out_slice[t_axis] = slice(in_len, in_len + out_len)
    return in_slice, out_slice

def change_layout_np(data,
                     in_layout='NHWT', out_layout='NHWT',
                     ret_contiguous=False):
    # first convert to 'NHWT'
    if in_layout == 'NHWT':
        pass
    elif in_layout == 'NTHW':
        data = np.transpose(data,
                            axes=(0, 2, 3, 1))
    elif in_layout == 'NWHT':
        data = np.transpose(data,
                            axes=(0, 2, 1, 3))
    elif in_layout == 'NTCHW':
        data = data[:, :, 0, :, :]
        data = np.transpose(data,
                            axes=(0, 2, 3, 1))
    elif in_layout == 'NTHWC':
        data = data[:, :, :, :, 0]
        data = np.transpose(data,
                            axes=(0, 2, 3, 1))
    elif in_layout == 'NTWHC':
        data = data[:, :, :, :, 0]
        data = np.transpose(data,
                            axes=(0, 3, 2, 1))
    elif in_layout == 'TNHW':
        data = np.transpose(data,
                            axes=(1, 2, 3, 0))
    elif in_layout == 'TNCHW':
        data = data[:, :, 0, :, :]
        data = np.transpose(data,
                            axes=(1, 2, 3, 0))
    else:
        raise NotImplementedError

    if out_layout == 'NHWT':
        pass
    elif out_layout == 'NTHW':
        data = np.transpose(data,
                            axes=(0, 3, 1, 2))
    elif out_layout == 'NWHT':
        data = np.transpose(data,
                            axes=(0, 2, 1, 3))
    elif out_layout == 'NTCHW':
        data = np.transpose(data,
                            axes=(0, 3, 1, 2))
        data = np.expand_dims(data, axis=2)
    elif out_layout == 'NTHWC':
        data = np.transpose(data,
                            axes=(0, 3, 1, 2))
        data = np.expand_dims(data, axis=-1)
    elif out_layout == 'NTWHC':
        data = np.transpose(data,
                            axes=(0, 3, 2, 1))
        data = np.expand_dims(data, axis=-1)
    elif out_layout == 'TNHW':
        data = np.transpose(data,
                            axes=(3, 0, 1, 2))
    elif out_layout == 'TNCHW':
        data = np.transpose(data,
                            axes=(3, 0, 1, 2))
        data = np.expand_dims(data, axis=2)
    else:
        raise NotImplementedError
    if ret_contiguous:
        data = data.ascontiguousarray()
    return data

def change_layout_torch(data,
                        in_layout='NHWT', out_layout='NHWT',
                        ret_contiguous=False):
    # first convert to 'NHWT'
    if in_layout == 'NHWT':
        pass
    elif in_layout == 'NTHW':
        data = data.permute(0, 2, 3, 1)
    elif in_layout == 'NTCHW':
        data = data[:, :, 0, :, :]
        data = data.permute(0, 2, 3, 1)
    elif in_layout == 'NTHWC':
        data = data[:, :, :, :, 0]
        data = data.permute(0, 2, 3, 1)
    elif in_layout == 'TNHW':
        data = data.permute(1, 2, 3, 0)
    elif in_layout == 'TNCHW':
        data = data[:, :, 0, :, :]
        data = data.permute(1, 2, 3, 0)
    else:
        raise NotImplementedError

    if out_layout == 'NHWT':
        pass
    elif out_layout == 'NTHW':
        data = data.permute(0, 3, 1, 2)
    elif out_layout == 'NTCHW':
        data = data.permute(0, 3, 1, 2)
        data = torch.unsqueeze(data, dim=2)
    elif out_layout == 'NTHWC':
        data = data.permute(0, 3, 1, 2)
        data = torch.unsqueeze(data, dim=-1)
    elif out_layout == 'TNHW':
        data = data.permute(3, 0, 1, 2)
    elif out_layout == 'TNCHW':
        data = data.permute(3, 0, 1, 2)
        data = torch.unsqueeze(data, dim=2)
    else:
        raise NotImplementedError
    if ret_contiguous:
        data = data.contiguous()
    return data
