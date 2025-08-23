# #----------------------------------------------------------------------------------------------------
# # Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# # Please check the licenses of the respective works utilized here before using this script.
# # 🖋️ Jishnu Jaykumar Padalunkal (2025).
# #----------------------------------------------------------------------------------------------------
# # Class for OCIDDataset
# #----------------------------------------------------------------------------------------------------

import os
import cv2
import torch
import numpy as np
from pathlib import Path
import open3d
import logging
from detectron2.structures import BoxMode
import pycocotools.mask as pycocotools_mask

from .base import BaseDataset
from .utils import augmentation, blob, mask as util_

logger = logging.getLogger(__name__)

class OCIDDataset(BaseDataset):
    OBJECTS_LABEL = 2
    
    def __init__(self, image_set="train", data_path=None, eval=False, config=None):
        super().__init__(image_set, data_path, eval, config)
        self._name = f'ocid_object_{image_set}'
        self.image_paths = self._list_dataset()
        logger.info(f'{len(self.image_paths)} images for dataset {self._name}')

    def _get_default_path(self):
        return Path(os.path.expanduser("~")) / 'data' / 'OCID'

    def _list_dataset(self):
        data_path = self._data_path
        seqs = sorted(list(data_path.glob('**/*seq*')))
        image_paths = []
        for seq in seqs:
            if self.image_set == 'train' and ('seq-01' in str(seq) or 'seq-02' in str(seq) or 'seq-03' in str(seq) or 'seq-04' in str(seq) or 'seq-05' in str(seq)):
                paths = sorted(list((seq / 'rgb').glob('*.png')))
                image_paths += paths
            elif self.image_set == 'test' and ('seq-06' in str(seq) or 'seq-07' in str(seq)):
                 paths = sorted(list((seq / 'rgb').glob('*.png')))
                 image_paths += paths
        return image_paths

    def __getitem__(self, idx):
        filename = str(self.image_paths[idx])
        im = cv2.imread(filename)
        if im is None: logger.error(f"Failed to load image: {filename}"); return None
        
        labels_filename = filename.replace('/rgb/', '/label/')
        foreground_labels = util_.imread_indexed(labels_filename)
        if foreground_labels is None: logger.warning(f"Missing mask: {labels_filename}"); return None
        foreground_labels[foreground_labels == 1] = 0
        if 'table' in labels_filename: foreground_labels[foreground_labels == 2] = 0
        
        xyz_img = None
        pcd_filename = filename.replace('/rgb/', '/pcd/').replace('.png', '.pcd')
        if os.path.exists(pcd_filename):
            try:
                pcd = open3d.io.read_point_cloud(pcd_filename)
                pcloud = np.asarray(pcd.points).astype(np.float32)
                pcloud[np.isnan(pcloud)] = 0
                xyz_img = pcloud.reshape((im.shape[0], im.shape[1], 3))
            except Exception as e:
                logger.warning(f"Failed to load PCD file {pcd_filename}: {e}"); return None
        
        im, foreground_labels, xyz_img, boxes, binary_masks, labels = self._apply_augmentations(im, foreground_labels, xyz_img)

        record = {
            "file_name": filename, "image_id": idx, "height": im.shape[0], "width": im.shape[1],
            "image_color": torch.from_numpy(im).permute(2, 0, 1).float(),
            "raw_depth": xyz_img,
            "depth": torch.from_numpy(xyz_img).permute(2, 0, 1) if xyz_img is not None else torch.zeros((3, im.shape[0], im.shape[1]))
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