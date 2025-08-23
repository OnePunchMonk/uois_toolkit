# #----------------------------------------------------------------------------------------------------
# # Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# # Please check the licenses of the respective works utilized here before using this script.
# # 🖋️ Jishnu Jaykumar Padalunkal (2025).
# #----------------------------------------------------------------------------------------------------
# # Class for RobotPushingDataset
# #----------------------------------------------------------------------------------------------------
import os
import cv2
import torch
import numpy as np
from pathlib import Path
import scipy.io
import logging
from detectron2.structures import BoxMode
import pycocotools.mask as pycocotools_mask

from .base import BaseDataset
from .utils import augmentation, blob

logger = logging.getLogger(__name__)

class RobotPushingDataset(BaseDataset):
    OBJECTS_LABEL = 1

    def __init__(self, image_set="train", data_path=None, eval=False, config=None):
        super().__init__(image_set, data_path, eval, config)
        self._name = f'robot_pushing_object_{image_set}'
        data_path_suffix = 'training_set' if self.image_set == 'train' else 'test_set'
        self._data_path = self._data_path / data_path_suffix
        self.image_paths = self._list_dataset()
        logger.info(f'{len(self.image_paths)} images for dataset {self._name}')

    def _get_default_path(self):
        return Path(os.path.expanduser("~")) / 'data' / 'pushing_data'

    def _list_dataset(self):
        seqs = sorted(list(self._data_path.glob('*T*')))
        image_paths = []
        for seq in seqs:
            paths = sorted(list((seq).glob('color*.jpg')))
            image_paths.extend(paths)
        return image_paths

    def __getitem__(self, idx):
        filename = str(self.image_paths[idx])
        im = cv2.imread(filename)
        if im is None: logger.error(f"Failed to load image: {filename}"); return None

        meta_filename = filename.replace('color', 'meta').replace('.jpg', '.mat')
        labels_filename = filename.replace('color', 'label-final').replace('.jpg', '.png')
        depth_filename = filename.replace('color', 'depth').replace('.jpg', '.png')

        try:
            meta_data = scipy.io.loadmat(meta_filename)
        except Exception as e: logger.warning(f"Failed to load meta file {meta_filename}: {e}"); return None

        foreground_labels = cv2.imread(labels_filename, cv2.IMREAD_GRAYSCALE)
        if foreground_labels is None: logger.warning(f"Missing mask: {labels_filename}"); return None
        
        depth_img = cv2.imread(depth_filename, cv2.IMREAD_ANYDEPTH)
        if depth_img is None: depth_img = np.zeros((im.shape[0], im.shape[1]), dtype=np.float32)
        
        xyz_img = self._compute_xyz_from_meta(depth_img.astype(np.float32), meta_data)
        im, foreground_labels, xyz_img, boxes, binary_masks, labels = self._apply_augmentations(im, foreground_labels, xyz_img, meta_data)

        record = {
            "file_name": filename, "image_id": idx, "height": im.shape[0], "width": im.shape[1],
            "image_color": torch.from_numpy(im).permute(2, 0, 1).float(),
            "raw_depth": xyz_img,
            "depth": torch.from_numpy(xyz_img).permute(2, 0, 1)
        }
        
        objs = []
        for i in range(boxes.shape[0]):
            mask_img = binary_masks[:, :, i]
            objs.append({
                "bbox": boxes[i].tolist(), "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": pycocotools_mask.encode(np.asfortranarray(mask_img)), "category_id": 1,
            })
        record["annotations"] = objs
        return record

    def _compute_xyz_from_meta(self, depth_img, meta_data):
        intrinsics = meta_data['intrinsic_matrix']
        factor_depth = meta_data['factor_depth']
        fx, fy, px, py = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
        height, width = depth_img.shape
        depth_img = depth_img / factor_depth
        indices = np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)
        z_e = depth_img
        x_e = (indices[..., 1] - px) * z_e / fx
        y_e = (indices[..., 0] - py) * z_e / fy
        xyz_img = np.stack([x_e, y_e, z_e], axis=-1)
        return xyz_img