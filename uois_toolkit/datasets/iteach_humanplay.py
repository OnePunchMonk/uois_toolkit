# #----------------------------------------------------------------------------------------------------
# # Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# # Please check the licenses of the respective works utilized here before using this script.
# # 🖋️ Jishnu Jaykumar Padalunkal (2025).
# #----------------------------------------------------------------------------------------------------
# # Class for iTeachHumanPlayDataset
# #----------------------------------------------------------------------------------------------------

import os
import cv2
import torch
import numpy as np
from pathlib import Path
import logging
from detectron2.structures import BoxMode
import pycocotools.mask as pycocotools_mask

from .base import BaseDataset
from .utils import augmentation, blob, mask as util_

logger = logging.getLogger(__name__)

class iTeachHumanPlayDataset(BaseDataset):
    OBJECTS_LABEL = 1

    def __init__(self, image_set="train", data_path=None, eval=False, config=None):
        super().__init__(image_set, data_path, eval, config)
        self._name = f'iteach_humanplay_object_{image_set}'
        self.image_paths = self._list_dataset()
        logger.info(f'{len(self.image_paths)} images for dataset {self._name}')

    def _get_default_path(self):
        return Path(os.path.expanduser("~")) / 'data' / 'iteach_humanplay_data'

    def _list_dataset(self):
        data_path = self._data_path
        seqs = sorted(list(data_path.glob("scene*")))
        image_paths = []
        for seq in seqs:
            paths = sorted(list((seq / 'rgb').glob('*.png')))
            image_paths.extend(paths)
        return image_paths

    def _get_intrinsic_matrix(self):
        return np.array([
            [527.8869068647631, 0.0, 321.7148665756361],
            [0.0, 524.7942507494529, 230.2819198622499],
            [0.0, 0.0, 1.0],
        ])

    def __getitem__(self, idx):
        filename = str(self.image_paths[idx])
        im = cv2.imread(filename)
        if im is None: logger.error(f"Failed to load image: {filename}"); return None

        labels_filename = filename.replace('rgb', 'gt_masks')
        foreground_labels = cv2.imread(labels_filename, cv2.IMREAD_GRAYSCALE)
        if foreground_labels is None: logger.warning(f"Missing mask: {labels_filename}"); return None

        depth_filename = filename.replace('rgb', 'depth')
        depth_img = cv2.imread(depth_filename, cv2.IMREAD_ANYDEPTH)
        if depth_img is None: depth_img = np.zeros((im.shape[0], im.shape[1]), dtype=np.float32)

        height, width = depth_img.shape
        intrinsics = self._get_intrinsic_matrix()
        fx, fy, px, py = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
        depth_img = depth_img / 1000.0
        
        indices = util_.build_matrix_of_indices(height, width)
        z_e = depth_img
        x_e = (indices[..., 1] - px) * z_e / fx
        y_e = (indices[..., 0] - py) * z_e / fy
        xyz_img = np.stack([x_e, y_e, z_e], axis=-1)

        im, foreground_labels, xyz_img, boxes, binary_masks, labels = self._apply_augmentations(im, foreground_labels, xyz_img)

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