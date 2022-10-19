from packaging import version
import torch


if version.parse(torch.__version__) >= version.parse('1.11.0'):
    # Starting from torch>=1.11.0, the attribute `optimizer` is set in SequentialLR. See https://github.com/pytorch/pytorch/pull/67406
    from torch.optim.lr_scheduler import SequentialLR
else:
    from torch.optim.lr_scheduler import _LRScheduler
    from bisect import bisect_right

    class SequentialLR(_LRScheduler):
        """Receives the list of schedulers that is expected to be called sequentially during
        optimization process and milestone points that provides exact intervals to reflect
        which scheduler is supposed to be called at a given epoch.

        Args:
            schedulers (list): List of chained schedulers.
            milestones (list): List of integers that reflects milestone points.

        Example:
            >>> # Assuming optimizer uses lr = 1. for all groups
            >>> # lr = 0.1     if epoch == 0
            >>> # lr = 0.1     if epoch == 1
            >>> # lr = 0.9     if epoch == 2
            >>> # lr = 0.81    if epoch == 3
            >>> # lr = 0.729   if epoch == 4
            >>> scheduler1 = ConstantLR(self.opt, factor=0.1, total_iters=2)
            >>> scheduler2 = ExponentialLR(self.opt, gamma=0.9)
            >>> scheduler = SequentialLR(self.opt, schedulers=[scheduler1, scheduler2], milestones=[2])
            >>> for epoch in range(100):
            >>>     train(...)
            >>>     validate(...)
            >>>     scheduler.step()
        """

        def __init__(self, optimizer, schedulers, milestones, last_epoch=-1, verbose=False):
            for scheduler_idx in range(1, len(schedulers)):
                if (schedulers[scheduler_idx].optimizer != schedulers[0].optimizer):
                    raise ValueError(
                        "Sequential Schedulers expects all schedulers to belong to the same optimizer, but "
                        "got schedulers at index {} and {} to be different".format(0, scheduler_idx)
                    )
            if (len(milestones) != len(schedulers) - 1):
                raise ValueError(
                    "Sequential Schedulers expects number of schedulers provided to be one more "
                    "than the number of milestone points, but got number of schedulers {} and the "
                    "number of milestones to be equal to {}".format(len(schedulers), len(milestones))
                )
            self.optimizer = optimizer
            self._schedulers = schedulers
            self._milestones = milestones
            self.last_epoch = last_epoch + 1

        def step(self):
            self.last_epoch += 1
            idx = bisect_right(self._milestones, self.last_epoch)
            if idx > 0 and self._milestones[idx - 1] == self.last_epoch:
                self._schedulers[idx].step(0)
            else:
                self._schedulers[idx].step()

        def state_dict(self):
            """Returns the state of the scheduler as a :class:`dict`.

            It contains an entry for every variable in self.__dict__ which
            is not the optimizer.
            The wrapped scheduler states will also be saved.
            """
            state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', '_schedulers')}
            state_dict['_schedulers'] = [None] * len(self._schedulers)

            for idx, s in enumerate(self._schedulers):
                state_dict['_schedulers'][idx] = s.state_dict()

            return state_dict

        def load_state_dict(self, state_dict):
            """Loads the schedulers state.

            Args:
                state_dict (dict): scheduler state. Should be an object returned
                    from a call to :meth:`state_dict`.
            """
            _schedulers = state_dict.pop('_schedulers')
            self.__dict__.update(state_dict)
            # Restore state_dict keys in order to prevent side effects
            # https://github.com/pytorch/pytorch/issues/32756
            state_dict['_schedulers'] = _schedulers

            for idx, s in enumerate(_schedulers):
                self._schedulers[idx].load_state_dict(s)

def warmup_lambda(warmup_steps, min_lr_ratio=0.1):
    def ret_lambda(epoch):
        if epoch <= warmup_steps:
            return min_lr_ratio + (1.0 - min_lr_ratio) * epoch / warmup_steps
        else:
            return 1.0
    return ret_lambda
