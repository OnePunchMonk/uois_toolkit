#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2025).
#----------------------------------------------------------------------------------------------------
# uois_toolkit/core/datasets/utils/blob.py

import numpy as np
import cv2

def chromatic_transform(im, label=None):
    """
    Applies chromatic (color-based) augmentations to an image.
    This includes random changes to brightness, contrast, and saturation.
    """
    im = im.copy()
    
    # --- Brightness ---
    # Randomly scale the brightness of the image
    brightness_scale = np.random.uniform(0.6, 1.4)
    im = im.astype(np.float32) * brightness_scale
    
    # --- Contrast ---
    # Randomly adjust the contrast
    contrast_scale = np.random.uniform(0.6, 1.4)
    im_mean = np.mean(im)
    im = (im - im_mean) * contrast_scale + im_mean
    
    # Clip values to be within the valid [0, 255] range
    im = np.clip(im, 0, 255)

    # --- Saturation ---
    # Convert to HSV color space to manipulate saturation
    im_hsv = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2HSV)
    
    # Randomly scale the saturation channel
    saturation_scale = np.random.uniform(0.6, 1.4)
    im_hsv[:, :, 1] = np.clip(im_hsv[:, :, 1].astype(np.float32) * saturation_scale, 0, 255)
    
    # Convert back to BGR color space
    im = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
    
    return im.astype(np.uint8)


def add_noise(im):
    """
    Adds Gaussian noise to an image.
    """
    im = im.copy()
    
    # Generate Gaussian noise with a random standard deviation
    # The standard deviation is chosen from a range to vary the noise level
    noise_std = np.random.randint(5, 25)
    
    # Create a noise matrix with the same dimensions as the image
    noise = np.random.normal(0, noise_std, im.shape)
    
    # Add the noise to the image and clip the values to the valid [0, 255] range
    noisy_im = np.clip(im.astype(np.float32) + noise, 0, 255)
    
    return noisy_im.astype(np.uint8)
