#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2025).
#----------------------------------------------------------------------------------------------------
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import logging
import torch

from .tabletop import TabletopDataset
from .ocid import OCIDDataset
from .osd import OSDDataset
from .robot_pushing import RobotPushingDataset
from .iteach_humanplay import iTeachHumanPlayDataset

logger = logging.getLogger(__name__)

DATASET_MAPPING = {
    "tabletop": TabletopDataset,
    "ocid": OCIDDataset,
    "osd": OSDDataset,
    "robot_pushing": RobotPushingDataset,
    "iteach_humanplay": iTeachHumanPlayDataset,
}

def uois_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    return None if not batch else torch.utils.data.dataloader.default_collate(batch)

class UOISDataModule(pl.LightningDataModule):
    def __init__(self, dataset_class, data_path, batch_size, num_workers, config=None):
        super().__init__()
        self.save_hyperparameters(ignore=['dataset_class', 'config'])
        self.dataset_class = dataset_class
        self.config = config

    def setup(self, stage=None):
        self.train_dataset = self.dataset_class(
            image_set='train', data_path=self.hparams.data_path, eval=False, config=self.config
        )
        self.val_dataset = self.dataset_class(
            image_set='test', data_path=self.hparams.data_path, eval=True, config=self.config
        )
        self.test_dataset = self.dataset_class(
            image_set='test', data_path=self.hparams.data_path, eval=True, config=self.config
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            collate_fn=uois_collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=uois_collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=uois_collate_fn
        )

def get_datamodule(
    dataset_name: str,
    data_path: str,
    batch_size: int = 4,
    num_workers: int = 4,
    config=None
) -> pl.LightningDataModule:
    dataset_name = dataset_name.lower()
    dataset_class = DATASET_MAPPING.get(dataset_name)
    if dataset_class is None:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Choose from {list(DATASET_MAPPING.keys())}"
        )
    
    return UOISDataModule(
        dataset_class=dataset_class,
        data_path=data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        config=config
    )