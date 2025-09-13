# #----------------------------------------------------------------------------------------------------
# # Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# # Please check the licenses of the respective works utilized here before using this script.
# # 🖋️ Jishnu Jaykumar Padalunkal (2025).
# #----------------------------------------------------------------------------------------------------
# # Class for TabletopObjectDataset
# #----------------------------------------------------------------------------------------------------

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from detectron2.structures import BoxMode
import pycocotools.mask as pycocotools_mask
import logging

from .base import BaseDataset
from .utils import mask as util_

logger = logging.getLogger(__name__)

class TabletopDataset(BaseDataset):
    OBJECTS_LABEL = 2
    
    def __init__(self, image_set="train", data_path=None, eval=False, config=None):
        super().__init__(image_set, data_path, eval, config)
        self._name = f'tabletop_object_{image_set}'
        self.image_paths = self._list_dataset()
        logger.info(f'{len(self.image_paths)} images for dataset {self._name}')

    def _get_default_path(self):
        return Path(os.path.expanduser("~")) / 'data' / 'tabletop'

    def _list_dataset(self):
        data_path_suffix = 'training_set' if self.image_set == 'train' else 'test_set'
        data_path = self._data_path / data_path_suffix
        return [p for seq in sorted(list(data_path.glob('scene_*'))) for p in sorted(list(seq.glob('rgb_*.jpeg')))]

    def _compute_xyz(self, depth_img, camera_params):
        if 'fx' in camera_params and 'fy' in camera_params:
            fx, fy, x_offset, y_offset = camera_params['fx'], camera_params['fy'], camera_params['x_offset'], camera_params['y_offset']
        else:
            aspect_ratio = camera_params['img_width'] / camera_params['img_height']
            e = 1 / np.tan(np.radians(camera_params['fov'] / 2.))
            t = camera_params['near'] / e
            b = -t
            r = t * aspect_ratio
            l = -r
            alpha = camera_params['img_width'] / (r - l)
            focal_length = camera_params['near'] * alpha
            fx, fy = focal_length, focal_length
            x_offset, y_offset = camera_params['img_width'] / 2, camera_params['img_height'] / 2

        indices = util_.build_matrix_of_indices(camera_params['img_height'], camera_params['img_width'])
        z_e = depth_img
        x_e = (indices[..., 1] - x_offset) * z_e / fx
        y_e = (indices[..., 0] - y_offset) * z_e / fy
        xyz_img = np.stack([x_e, y_e, z_e], axis=-1)
        return xyz_img

    def __getitem__(self, idx):
        filename = str(self.image_paths[idx])
        im = cv2.imread(filename)
        if im is None: logger.error(f"Failed to load image: {filename}"); return None

        labels_filename = filename.replace('rgb_', 'segmentation_')
        foreground_labels = util_.imread_indexed(labels_filename)
        if foreground_labels is None: logger.warning(f"Missing mask: {labels_filename}"); return None
        foreground_labels[foreground_labels == 1] = 0

        depth_img = cv2.imread(filename.replace('rgb_', 'depth_'), cv2.IMREAD_ANYDEPTH)
        if depth_img is None: depth_img = np.zeros((im.shape[0], im.shape[1]), dtype=np.float32)
        depth_img = (depth_img / 1000.0).astype(np.float32)
        xyz_img = self._compute_xyz(depth_img, self.data_loading_params)
        
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