#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2025).
#----------------------------------------------------------------------------------------------------

# import numpy as np

# def mask_to_tight_box(mask):
#     a = np.transpose(np.nonzero(mask))
#     if len(a) == 0:
#         return 0, 0, 0, 0
#     return np.min(a[:, 1]), np.min(a[:, 0]), np.max(a[:, 1]), np.max(a[:, 0])

# uois_toolkit/core/datasets/utils/mask.py

import numpy as np
import cv2

def build_matrix_of_indices(height, width):
    return np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)

def mask_to_tight_box(mask):
    a = np.transpose(np.nonzero(mask))
    if a.size == 0:
        return [0, 0, 0, 0]
    bbox = np.min(a[:, 1]), np.min(a[:, 0]), np.max(a[:, 1]), np.max(a[:, 0])
    return bbox
    
def imread_indexed(filename):
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)