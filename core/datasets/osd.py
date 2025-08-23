# #----------------------------------------------------------------------------------------------------
# # Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# # Please check the licenses of the respective works utilized here before using this script.
# # 🖋️ Jishnu Jaykumar Padalunkal (2025).
# #----------------------------------------------------------------------------------------------------
# # Class for OSDDataset
# #----------------------------------------------------------------------------------------------------

import os
import cv2
import torch
import numpy as np
import glob
from pathlib import Path
import logging
import pycocotools.mask as pycocotools_mask
from detectron2.structures import BoxMode
import imageio

from .base import BaseDataset
from .utils import mask as util_

logger = logging.getLogger(__name__)

def normalize_depth(depth, min_val=250.0, max_val=1500.0):
    depth[depth < min_val] = min_val
    depth[depth > max_val] = max_val
    depth = (depth - min_val) / (max_val - min_val) * 255
    depth = np.expand_dims(depth, -1)
    depth = np.uint8(np.repeat(depth, 3, -1))
    return depth

def inpaint_depth(depth, factor=1, kernel_size=3, dilate=False):
    H, W, _ = depth.shape
    resized_depth = cv2.resize(depth, (W // factor, H // factor))
    mask_img = np.all(resized_depth == 0, axis=2).astype(np.uint8)
    if dilate: mask_img = cv2.dilate(mask_img, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
    inpainted_data = cv2.inpaint(resized_depth, mask_img, kernel_size, cv2.INPAINT_TELEA)
    inpainted_data = cv2.resize(inpainted_data, (W, H))
    depth = np.where(depth == 0, inpainted_data, depth)
    return depth

class OSDDataset(BaseDataset):
    OBJECTS_LABEL = 1

    def __init__(self, image_set="train", data_path=None, eval=False, config=None):
        super().__init__(image_set, data_path, eval, config)
        self._name = f'osd_object_{image_set}'
        self.image_paths = self._list_dataset()
        logger.info(f'{len(self.image_paths)} images for dataset {self._name}')
        if not os.path.exists(self._data_path):
            raise FileNotFoundError(f'OSD path does not exist: {self._data_path}')

    def _get_default_path(self):
        return Path(os.path.expanduser("~")) / 'data' / 'OSD'

    def _list_dataset(self):
        data_path = Path(self._data_path)
        rgb_paths = sorted(list(data_path.glob('image_color/*.png')))
        return rgb_paths

    def __getitem__(self, idx):
        filename = str(self.image_paths[idx])
        im = cv2.imread(filename)
        if im is None: logger.error(f"Failed to load image: {filename}"); return None

        labels_filename = filename.replace('image_color', 'annotation')
        foreground_labels = util_.imread_indexed(labels_filename)
        if foreground_labels is None: logger.warning(f"Missing mask: {labels_filename}"); return None

        disparity_path = filename.replace('image_color', 'disparity')
        disparity_img = imageio.imread(disparity_path)
        normalized_disparity = normalize_depth(disparity_img)
        inpainted_disparity = inpaint_depth(normalized_disparity)
        
        im, foreground_labels, _, boxes, binary_masks, labels = self._apply_augmentations(im, foreground_labels, None)

        record = {
            "file_name": filename, "image_id": idx, "height": im.shape[0], "width": im.shape[1],
            "image_color": torch.from_numpy(im).permute(2, 0, 1).float(),
            "depth": torch.from_numpy(inpainted_disparity).permute(2, 0, 1).float() / 255.0,
            "raw_depth": inpainted_disparity
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