# ----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2025).
# ----------------------------------------------------------------------------------------------------

from .datasets.base import BaseUOISDataset
from .datasets.iteach_humanplay import iTeachHumanPlayDataset
from .datasets.ocid import OCIDDataset
from .datasets.osd import OSDDataset
from .datasets.robot_pushing import RobotPushingDataset
from .datasets.tabletop import TabletopDataset

def get_dataset(dataset_name, image_set="train", data_path=None, eval=False, config=None):
    dataset_name = dataset_name.lower()
    datasets = {
        'iteach_humanplay': iTeachHumanPlayDataset,
        'ocid': OCIDDataset,
        'osd': OSDDataset,
        'robot_pushing': RobotPushingDataset,
        'tabletop': TabletopDataset
    }
    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(datasets.keys())}")
    return datasets[dataset_name](image_set=image_set, data_path=data_path, eval=eval, config=config)