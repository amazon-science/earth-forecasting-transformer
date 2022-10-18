import torch
from torch import nn
from torch.nn import functional as F


class Persistence(nn.Module):

    def __init__(self, layout="NTHWC"):
        super(Persistence, self).__init__()
        self.layout = layout
        self.t_axis = self.layout.find("T")
        self.in_seq_slice = [slice(None, None), ] * len(self.layout)
        self.in_seq_slice[self.t_axis] = slice(-1, None)

    def forward(self, in_seq, out_seq):
        out_len = out_seq.shape[self.t_axis]
        output = in_seq[self.in_seq_slice]
        output = torch.repeat_interleave(output,
                                         repeats=out_len,
                                         dim=self.t_axis)
        loss = F.mse_loss(output, out_seq)
        return output, loss
