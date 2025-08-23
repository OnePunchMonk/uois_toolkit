#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2025).
#----------------------------------------------------------------------------------------------------
# Config for UOIS Datasets
#----------------------------------------------------------------------------------------------------

# import numpy as np

# import numpy as np

# config = {
#     'PIXEL_MEANS': np.array([102.9801, 115.9465, 122.7717]),  # BGR
#     'INPUT': 'RGBD',
#     'MODE': 'TRAIN',
#     'FLOW_HEIGHT': 480,
#     'FLOW_WIDTH': 640,
#     'TRAIN': {
#         'CHROMATIC': True,
#         'ADD_NOISE': True,
#         'SYN_CROP': True,
#         'SYN_CROP_SIZE': 224,
#         'EMBEDDING_SAMPLING': True,
#         'EMBEDDING_SAMPLING_NUM': 1000,
#         'min_pixels': 200,
#         'max_pixels': 10000,
#         'min_padding_percentage': 0.05,
#         'max_padding_percentage': 0.5,
#         'CLASSES': (0, 1),
#     }
# }

import numpy as np

class Config:
    """
    A configuration class to hold all dataset and training parameters.
    This provides a single, centralized location for all settings.
    """
    def __init__(self):
        # --- Top-Level Parameters ---
        
        # Mean pixel values (BGR) for image normalization.
        self.PIXEL_MEANS = np.array([102.9801, 115.9465, 122.7717])
        
        # Defines the type of input data for the model (e.g., 'RGBD', 'COLOR').
        self.INPUT = 'RGBD'
        
        # Sets the operational mode ('TRAIN' or 'TEST'), which can affect augmentations.
        self.MODE = 'TRAIN'
        
        # Default dimensions for image and data processing.
        self.FLOW_HEIGHT = 480
        self.FLOW_WIDTH = 640

        # --- Training-Specific Parameters ---
        
        self.TRAIN = {
            # --- Augmentation Flags ---
            'CHROMATIC': True,  # Enable/disable chromatic (color) augmentation.
            'ADD_NOISE': True,  # Enable/disable adding noise to images.
            
            # --- Synthetic Cropping ---
            'SYN_CROP': True,  # Enable/disable synthetic cropping around objects.
            'SYN_CROP_SIZE': 224,  # The target size for cropped images.
            
            # --- Pixel Sampling ---
            'EMBEDDING_SAMPLING': True,  # Enable/disable sampling pixels from masks.
            'EMBEDDING_SAMPLING_NUM': 1000,  # Number of pixels to sample per instance.
            
            # --- Object Filtering ---
            'min_pixels': 200,  # Minimum number of pixels for an object to be included.
            'max_pixels': 10000,  # Maximum number of pixels for an object to be included.
            
            # --- Padding for Cropping ---
            'min_padding_percentage': 0.05,  # Minimum padding around an object during cropping.
            'max_padding_percentage': 0.5,   # Maximum padding around an object during cropping.
            
            # --- Class Definitions ---
            'CLASSES': (0, 1),  # Defines the classes (e.g., background, foreground).
        }

# Create a single, global instance of the configuration object.
cfg = Config()
