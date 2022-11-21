# Find the original code and discussion at https://github.com/PyTorchLightning/pytorch-lightning/discussions/10922
# We will need to use the AMP implementation from apex because https://discuss.pytorch.org/t/using-torch-utils-checkpoint-checkpoint-with-dataparallel/78452

from apex.parallel import DistributedDataParallel
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.overrides.base import (
    _LightningModuleWrapperBase,
    _LightningPrecisionModuleWrapperBase,
)

def unwrap_lightning_module(wrapped_model):
    model = wrapped_model
    if isinstance(model, DistributedDataParallel):
        model = unwrap_lightning_module(model.module)
    if isinstance(
        model, (_LightningModuleWrapperBase, _LightningPrecisionModuleWrapperBase)
    ):
        model = unwrap_lightning_module(model.module)
    return model


class ApexDDPStrategy(DDPStrategy):
    def _setup_model(self, model):
        return DistributedDataParallel(model, delay_allreduce=False)

    @property
    def lightning_module(self):
        return unwrap_lightning_module(self._model)


if __name__ == "__main__":
    # Correct usage of apex DDP, which can avoid error caused by using `torch.utils.checkpoint`
    # when using `strategy="ddp"` in pl.
    import pytorch_lightning as pl
    trainer = pl.Trainer(
        strategy=ApexDDPStrategy(find_unused_parameters=False, delay_allreduce=True),  # "ddp",
    )
