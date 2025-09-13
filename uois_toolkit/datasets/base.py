#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2025).
#----------------------------------------------------------------------------------------------------
# Base class for UOIS datasets
# This class provides a common interface and basic functionality for UOIS datasets.
# It handles image loading, data augmentation, and annotation processing.
#----------------------------------------------------------------------------------------------------

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from detectron2.structures import BoxMode
import pycocotools.mask as pycocotools_mask
import logging
import random

from ..config.config import cfg as default_cfg
from .utils import augmentation, blob, mask as util_

logger = logging.getLogger(__name__)

class BaseDataset(Dataset):
    BACKGROUND_LABEL, TABLE_LABEL, OBJECTS_LABEL = 0, 1, 2

    def __init__(self, image_set="train", data_path=None, eval=False, config=None):
        self.cfg = config if config is not None else default_cfg
        self.image_set = image_set
        self.eval = eval
        self.data_loading_params = {
            'img_width': self.cfg.FLOW_WIDTH, 'img_height': self.cfg.FLOW_HEIGHT,
            'near': 0.01, 'far': 100, 'fov': 45,
            'use_data_augmentation': self.cfg.TRAIN['CHROMATIC'] or self.cfg.TRAIN['ADD_NOISE'],
            'min_pixels': self.cfg.TRAIN['min_pixels'], 'max_pixels': self.cfg.TRAIN['max_pixels'],
        }
        self._data_path = Path(data_path) if data_path else self._get_default_path()

    def __len__(self):
        return len(self.image_paths)

    def _get_default_path(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def _list_dataset(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def process_label_to_annos(self, labels):
        H, W = labels.shape
        unique_labels = np.unique(labels)
        if unique_labels[0] == 0: unique_labels = unique_labels[1:]
        num_instances = unique_labels.shape[0]

        binary_masks = np.zeros((H, W, num_instances), dtype=np.float32)
        for i, label in enumerate(unique_labels):
            binary_masks[..., i] = (labels == label).astype(np.float32)
        
        boxes = np.zeros((num_instances, 4))
        for i in range(num_instances):
            boxes[i, :] = np.array(util_.mask_to_tight_box(binary_masks[..., i]))
        
        return boxes, binary_masks, unique_labels

    def _apply_augmentations(self, im, foreground_labels, xyz_img, meta_data=None):
        if not self.eval:
            if self.cfg.TRAIN['SYN_CROP']:
                im, foreground_labels, xyz_img = self._pad_crop_resize(im, foreground_labels, xyz_img)
            if self.cfg.TRAIN['EMBEDDING_SAMPLING']:
                foreground_labels = self._sample_pixels(foreground_labels, self.cfg.TRAIN['EMBEDDING_SAMPLING_NUM'])
            if self.cfg.TRAIN['CHROMATIC'] and random.random() > 0.1: im = blob.chromatic_transform(im)
            if self.cfg.TRAIN['ADD_NOISE'] and random.random() > 0.1: im = blob.add_noise(im)
            
        boxes, binary_masks, labels = self.process_label_to_annos(foreground_labels)
        
        return im, foreground_labels, xyz_img, boxes, binary_masks, labels

    def _pad_crop_resize(self, img, label, depth):
        H, W, _ = img.shape
        K = np.max(label)
        
        for _ in range(10):
            idx = np.random.randint(1, K + 1) if K > 0 else 0
            foreground = (label == idx).astype(np.float32)
            if np.sum(foreground) == 0: continue

            x_min, y_min, x_max, y_max = util_.mask_to_tight_box(foreground)
            cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
            x_delta, y_delta = x_max - x_min, y_max - y_min
            
            if x_delta > y_delta: y_min, y_max = cy - x_delta / 2, cy + x_delta / 2
            else: x_min, x_max = cx - y_delta / 2, cx + y_delta / 2
            
            sidelength = x_max - x_min
            padding = int(round(sidelength * random.uniform(self.cfg.TRAIN['min_padding_percentage'], self.cfg.TRAIN['max_padding_percentage']))) or 25
            
            x_min = max(int(x_min - padding), 0)
            x_max = min(int(x_max + padding), W - 1)
            y_min = max(int(y_min - padding), 0)
            y_max = min(int(y_max + padding), H - 1)
            
            if y_min >= y_max or x_min >= x_max: continue
            
            img_crop = img[y_min:y_max+1, x_min:x_max+1]
            label_crop = label[y_min:y_max+1, x_min:x_max+1]
            depth_crop = depth[y_min:y_max+1, x_min:x_max+1] if depth is not None else None

            s = self.cfg.TRAIN['SYN_CROP_SIZE']
            img_crop = cv2.resize(img_crop, (s, s))
            label_crop = cv2.resize(label_crop, (s, s), interpolation=cv2.INTER_NEAREST)
            if depth_crop is not None:
                depth_crop = cv2.resize(depth_crop, (s, s), interpolation=cv2.INTER_NEAREST)
            return img_crop, label_crop, depth_crop
        
        return img, label, depth

    def _sample_pixels(self, labels, num=1000):
        labels_new = np.full_like(labels, -1, dtype=np.int32)
        K = np.max(labels)
        for i in range(K + 1):
            index = np.where(labels == i)
            n = len(index[0])
            if n <= num:
                labels_new[index[0], index[1]] = i
            else:
                perm = np.random.permutation(n)
                selected = perm[:num]
                labels_new[index[0][selected], index[1][selected]] = i
        return labels_new
    
    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses must implement __getitem__ to load data.")