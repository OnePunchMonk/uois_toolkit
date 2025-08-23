# ----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2025).
# ----------------------------------------------------------------------------------------------------

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
import logging
import os
import numpy as np
import copy

# Import the main entry point and utilities from your library
from uois_toolkit import get_datamodule, cfg
from uois_toolkit.core.datasets.utils import set_seeds

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Dummy LightningModule for Comprehensive Testing ---
class DummyModel(pl.LightningModule):
    """A simple dummy model to test the data loading pipeline for all splits."""
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(3 * cfg.FLOW_HEIGHT * cfg.FLOW_WIDTH, 10)

    def _common_step(self, batch, batch_idx, stage):
        if batch is None:
            logger.warning(f"[{stage}] Skipping an invalid batch at index {batch_idx}.")
            return None
            
        images = batch.get("image_color")
        annotations = batch.get("annotations")
        
        if images is None:
            logger.error(f"[{stage}] Batch is missing 'image_color' key.")
            return None

        # This check is crucial for validating the annotation pipeline
        if not annotations or not annotations.get('bbox'):
             logger.warning(f"[{stage}] Batch {batch_idx} has no bounding box annotations.")
        else:
             logger.info(f"[{stage}] Batch {batch_idx}: Image shape: {images.shape}, BBox count: {len(annotations.get('bbox', []))}")

        batch_size, C, H, W = images.shape
        if self.layer.in_features != C * H * W:
            self.layer = torch.nn.Linear(C * H * W, 10).to(self.device)
        
        flat_images = images.view(batch_size, -1)
        output = self.layer(flat_images)
        loss = F.mse_loss(output, torch.rand_like(output))
        
        self.log(f'{stage}_loss', loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

# --- Detailed Validation Functions ---

def validate_seeding(dataset_name, data_path, config):
    """Verify that setting a seed produces reproducible data batches."""
    logger.info("--- Validating Seeding ---")
    
    set_seeds(42)
    dm1 = get_datamodule(dataset_name, data_path, batch_size=2, num_workers=0, config=config)
    dm1.setup()
    batch1 = next(iter(dm1.train_dataloader()))

    set_seeds(42)
    dm2 = get_datamodule(dataset_name, data_path, batch_size=2, num_workers=0, config=config)
    dm2.setup()
    batch2 = next(iter(dm2.train_dataloader()))

    if batch1 is not None and batch2 is not None and torch.equal(batch1["image_color"], batch2["image_color"]):
        logger.info("Seeding validation PASSED.")
    else:
        logger.error("Seeding validation FAILED: Batches are not identical.")

def validate_augmentations(dataset_name, data_path):
    """Verify that augmentation flags correctly modify the output data."""
    logger.info("--- Validating Augmentations ---")
    
    cfg_no_aug = copy.deepcopy(cfg)
    cfg_no_aug.TRAIN['CHROMATIC'] = False
    cfg_no_aug.TRAIN['ADD_NOISE'] = False
    cfg_no_aug.TRAIN['SYN_CROP'] = False
    
    dm_no_aug = get_datamodule(dataset_name, data_path, batch_size=1, num_workers=0, config=cfg_no_aug)
    dm_no_aug.setup()
    original_batch = next(iter(dm_no_aug.train_dataloader()))
    if original_batch is None:
        logger.error("Augmentation validation FAILED: Could not load initial batch.")
        return
    original_image = original_batch["image_color"]

    cfg_with_aug = copy.deepcopy(cfg)
    cfg_with_aug.TRAIN['CHROMATIC'] = True
    cfg_with_aug.TRAIN['ADD_NOISE'] = True
    cfg_with_aug.TRAIN['SYN_CROP'] = True

    dm_with_aug = get_datamodule(dataset_name, data_path, batch_size=1, num_workers=0, config=cfg_with_aug)
    dm_with_aug.setup()
    augmented_batch = next(iter(dm_with_aug.train_dataloader()))
    if augmented_batch is None:
        logger.error("Augmentation validation FAILED: Could not load augmented batch.")
        return
    augmented_image = augmented_batch["image_color"]

    if not torch.equal(original_image, augmented_image):
        logger.info("Chromatic/Noise augmentation validation PASSED.")
    else:
        logger.error("Chromatic/Noise augmentation validation FAILED: Image was not modified.")
        
    expected_size = cfg_with_aug.TRAIN['SYN_CROP_SIZE']
    if augmented_image.shape[2] == expected_size and augmented_image.shape[3] == expected_size:
        logger.info(f"SYN_CROP validation PASSED. Image resized to {expected_size}x{expected_size}.")
    else:
        logger.error(f"SYN_CROP validation FAILED: Expected size {expected_size}, got {augmented_image.shape[2:]}.")

def validate_depth_processing(dm):
    """Verify that depth data is being loaded."""
    logger.info("--- Validating Depth Processing ---")
    batch = next(iter(dm.train_dataloader()))
    if batch is not None and 'depth' in batch and batch['depth'].nelement() > 0 and batch['depth'].abs().sum() > 0:
        logger.info("Depth processing validation PASSED.")
    else:
        logger.error("Depth processing validation FAILED: 'depth' key missing, empty, or all zeros.")

# --- Main Test Function ---
def main():
    """Main function to run a comprehensive test suite for all datasets in the toolkit."""
    dataset_paths = {
        'tabletop': "/path/to/your/tabletop/dataset",
        'ocid': "/path/to/your/ocid/dataset",
        'osd': "/path/to/your/osd/dataset",
        'robot_pushing': "/path/to/your/robot_pushing/dataset",
        'iteach_humanplay': "/path/to/your/iteach_humanplay/dataset",
    }
    
    custom_config = copy.deepcopy(cfg)
    custom_config.TRAIN['CHROMATIC'] = False
    custom_config.TRAIN['ADD_NOISE'] = False
    custom_config.TRAIN['SYN_CROP'] = False
    logger.info("Defined a custom configuration with all augmentations turned OFF for the final pipeline test.")
    
    for name, path in dataset_paths.items():
        logger.info(f"\n{'='*25} Testing Dataset: {name.upper()} {'='*25}")
        
        if "/path/to/your" in path or not os.path.exists(path):
            logger.warning(f"Skipping {name.upper()}: Path is a placeholder or does not exist. Please update '{path}'.")
            continue
            
        try:
            # Step 1: Validate Seeding, Augmentations, and Depth with default config
            validate_seeding(name, path, cfg)
            validate_augmentations(name, path)
            dm_default = get_datamodule(name, path, batch_size=2, num_workers=0, config=cfg)
            dm_default.setup()
            validate_depth_processing(dm_default)

            # Step 2: Validate Full Dataloader Pipeline with a CUSTOM config
            logger.info("--- Validating Dataloader Splits (Train, Val, Test) with CUSTOM config ---")
            data_module = get_datamodule(name, path, batch_size=2, num_workers=2, config=custom_config)
            
            model = DummyModel()
            trainer = pl.Trainer(
                max_epochs=1, accelerator="auto", devices="auto",
                limit_train_batches=2, limit_val_batches=2, limit_test_batches=2,
                logger=False, enable_checkpointing=False,
            )
            
            logger.info("Testing trainer.fit() (uses train and val dataloaders)...")
            trainer.fit(model, datamodule=data_module)
            
            logger.info("Testing trainer.test() (uses test dataloader)...")
            trainer.test(model, datamodule=data_module)
            
            logger.info(f"--- Full Pipeline Test for {name.upper()} PASSED ---")

        except Exception as e:
            logger.error(f"--- Test for {name.upper()} FAILED ---", exc_info=True)

if __name__ == '__main__':
    main()
