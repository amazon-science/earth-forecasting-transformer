from pytorch_lightning import LightningDataModule
from ...config import cfg

class IMSLightningDataModule(LightningDataModule):
    def __init__(self):
        pass
        super(IMSLightningDataModule, self).__init__()

    def prepare_data(self) -> None:
        #download the data
        pass

    def setup(self, stage=None) -> None:
        #read https://lightning.ai/docs/pytorch/stable/data/datamodule.html how it is supposed to look
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        pass
