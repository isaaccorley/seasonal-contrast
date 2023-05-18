from torch.utils.data import DataLoader
from torchvision import transforms as T
from pytorch_lightning import LightningDataModule

from datasets.eurosat_dataset import EurosatDataset


class EurosatDataModule(LightningDataModule):

    def __init__(self, data_dir, batch_size=32, num_workers=8, transforms=T.ToTensor()):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms

    @property
    def num_classes(self):
        return 10

    def setup(self, stage=None):
        self.train_dataset = EurosatDataset(self.data_dir, split='train', transform=self.transforms)
        self.val_dataset = EurosatDataset(self.data_dir, split='val', transform=self.transforms)
        self.test_dataset = EurosatDataset(self.data_dir, split='test', transform=self.transforms)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=8,
            drop_last=False,
            pin_memory=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=8,
            drop_last=False,
            pin_memory=False
        )
