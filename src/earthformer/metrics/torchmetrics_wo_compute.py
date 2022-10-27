from typing import Any
import torch
from torchmetrics import Metric
from torchmetrics.utilities.exceptions import TorchMetricsUserError


class MetricsUpdateWithoutCompute(Metric):
    r"""
    Delete `batch_val = self.compute()` in `forward()` to reduce unnecessary computation
    """

    def __init__(self, **kwargs: Any):
        super(MetricsUpdateWithoutCompute, self).__init__(**kwargs)

    @torch.jit.unused
    def forward(self, *args: Any, **kwargs: Any):
        """``forward`` serves the dual purpose of both computing the metric on the current batch of inputs but also
        add the batch statistics to the overall accumululating metric state.

        Input arguments are the exact same as corresponding ``update`` method. The returned output is the exact same as
        the output of ``compute``.
        """
        # check if states are already synced
        if self._is_synced:
            raise TorchMetricsUserError(
                "The Metric shouldn't be synced when performing ``forward``. "
                "HINT: Did you forget to call ``unsync`` ?."
            )

        if self.full_state_update or self.full_state_update is None or self.dist_sync_on_step:
            self._forward_full_state_update_without_compute(*args, **kwargs)
        else:
            self._forward_reduce_state_update_without_compute(*args, **kwargs)

    def _forward_full_state_update_without_compute(self, *args: Any, **kwargs: Any):
        """forward computation using two calls to `update` to calculate the metric value on the current batch and
        accumulate global state.

        Doing this secures that metrics that need access to the full metric state during `update` works as expected.
        """
        # global accumulation
        self.update(*args, **kwargs)
        _update_count = self._update_count

        self._to_sync = self.dist_sync_on_step  # type: ignore
        # skip restore cache operation from compute as cache is stored below.
        self._should_unsync = False
        # skip computing on cpu for the batch
        _temp_compute_on_cpu = self.compute_on_cpu
        self.compute_on_cpu = False

        # save context before switch
        cache = {attr: getattr(self, attr) for attr in self._defaults}

        # call reset, update, compute, on single batch
        self._enable_grad = True  # allow grads for batch computation
        self.reset()
        self.update(*args, **kwargs)

        # restore context
        for attr, val in cache.items():
            setattr(self, attr, val)
        self._update_count = _update_count

        # restore context
        self._is_synced = False
        self._should_unsync = True
        self._to_sync = self.sync_on_compute
        self._computed = None
        self._enable_grad = False
        self.compute_on_cpu = _temp_compute_on_cpu

    def _forward_reduce_state_update_without_compute(self, *args: Any, **kwargs: Any):
        """forward computation using single call to `update` to calculate the metric value on the current batch and
        accumulate global state.

        This can be done when the global metric state is a sinple reduction of batch states.
        """
        # store global state and reset to default
        global_state = {attr: getattr(self, attr) for attr in self._defaults.keys()}
        _update_count = self._update_count
        self.reset()

        # local synchronization settings
        self._to_sync = self.dist_sync_on_step
        self._should_unsync = False
        _temp_compute_on_cpu = self.compute_on_cpu
        self.compute_on_cpu = False
        self._enable_grad = True  # allow grads for batch computation

        # calculate batch state and compute batch value
        self.update(*args, **kwargs)

        # reduce batch and global state
        self._update_count = _update_count + 1
        with torch.no_grad():
            self._reduce_states(global_state)

        # restore context
        self._is_synced = False
        self._should_unsync = True
        self._to_sync = self.sync_on_compute
        self._computed = None
        self._enable_grad = False
        self.compute_on_cpu = _temp_compute_on_cpu
