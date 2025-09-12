# # ----------------------------------------------------------------------------------------------------
# # Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# # Please check the licenses of the respective works utilized here before using this script.
# # 🖋️ Jishnu Jaykumar Padalunkal (2025).
# # ----------------------------------------------------------------------------------------------------

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
import logging
import os
import numpy as np
import copy
import pytest

# Import the main entry point and utilities from your library
from uois_toolkit import get_datamodule, cfg
from uois_toolkit.core.datasets.utils import set_seeds
from uois_toolkit.core.metrics import (
    precision,
    recall,
    f1_score,
    intersection_over_union,
    iou_threshold,
    compute_metrics,
    get_available_metrics,
)

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------
# Pytest Fixtures for Command-Line Path Configuration
# ---------------------------------------------------

def pytest_addoption(parser):
    """Adds the --dataset_path option to the pytest command line."""
    parser.addoption(
        "--dataset_path",
        action="append",
        default=[],
        help="Specify dataset paths in the format name=path (e.g., tabletop=/data/tabletop)",
    )

@pytest.fixture(scope="session")
def dataset_paths(request):
    """Parses the --dataset_path arguments and returns a dictionary."""
    path_dict = {}
    for arg in request.config.getoption("--dataset_path"):
        try:
            name, path = arg.split("=", 1)
            if os.path.exists(path):
                path_dict[name] = path
            else:
                logger.warning(f"Path for dataset '{name}' does not exist: {path}. It will be skipped.")
        except ValueError:
            pytest.fail(f"Invalid format for --dataset_path: '{arg}'. Use 'name=path'.")
    return path_dict

# ---------------------------------------------------
# Dummy LightningModule for Testing
# ---------------------------------------------------

class DummyModel(pl.LightningModule):
    """A simple dummy model to test the data loading pipeline for all splits."""
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(3 * cfg.FLOW_HEIGHT * cfg.FLOW_WIDTH, 10)

    def _common_step(self, batch, batch_idx, stage):
        if batch is None: return None
        images = batch.get("image_color")
        batch_size, C, H, W = images.shape
        if self.layer.in_features != C * H * W:
            self.layer = torch.nn.Linear(C * H * W, 10).to(self.device)
        
        flat_images = images.view(batch_size, -1)
        output = self.layer(flat_images)
        loss = F.mse_loss(output, torch.rand_like(output))
        self.log(f'{stage}_loss', loss)
        return loss

    def training_step(self, batch, batch_idx): return self._common_step(batch, batch_idx, "train")
    def validation_step(self, batch, batch_idx): return self._common_step(batch, batch_idx, "val")
    def test_step(self, batch, batch_idx): return self._common_step(batch, batch_idx, "test")
    def configure_optimizers(self): return torch.optim.Adam(self.parameters(), lr=1e-4)

# ---------------------------------------------------
# Main Test Suite
# ---------------------------------------------------

@pytest.mark.parametrize("dataset_name", ["tabletop", "ocid", "osd", "robot_pushing", "iteach_humanplay"])
def test_dataset_pipeline(dataset_name, dataset_paths):
    """
    A single, parameterized test that validates the entire pipeline for each dataset.
    """
    if dataset_name not in dataset_paths:
        pytest.skip(f"Path for dataset '{dataset_name}' not provided via --dataset_path. Skipping.")

    path = dataset_paths[dataset_name]
    logger.info(f"\n{'='*25} Testing Dataset: {dataset_name.upper()} {'='*25}")

    # --- 1. Validate Seeding ---
    _validate_seeding(dataset_name, path)

    # --- 2. Validate Augmentations ---
    _validate_augmentations(dataset_name, path)

    # --- 3. Validate Full Dataloader Pipeline ---
    logger.info("--- Validating Dataloader Splits (Train, Val, Test) ---")
    data_module = get_datamodule(dataset_name, path, batch_size=2, num_workers=2, config=cfg)
    model = DummyModel()
    trainer = pl.Trainer(
        max_epochs=1, accelerator="auto", devices="auto",
        limit_train_batches=2, limit_val_batches=2, limit_test_batches=2,
        logger=False, enable_checkpointing=False,
    )
    
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
    logger.info(f"--- Full Pipeline Test for {dataset_name.upper()} PASSED ---")


# ---------------------------------------------------
# Helper Validation Functions
# ---------------------------------------------------

def _validate_seeding(dataset_name, data_path):
    """Helper to verify that seeding produces reproducible batches."""
    logger.info("--- Validating Seeding ---")
    set_seeds(42)
    dm1 = get_datamodule(dataset_name, data_path, batch_size=1, num_workers=0, config=cfg)
    dm1.setup()
    batch1 = next(iter(dm1.train_dataloader()))

    set_seeds(42)
    dm2 = get_datamodule(dataset_name, data_path, batch_size=1, num_workers=0, config=cfg)
    dm2.setup()
    batch2 = next(iter(dm2.train_dataloader()))
    
    assert batch1 is not None and batch2 is not None, "Failed to load batches for seeding test."
    assert torch.equal(batch1["image_color"], batch2["image_color"]), "Seeding validation FAILED."
    logger.info("Seeding validation PASSED.")

def _validate_augmentations(dataset_name, data_path):
    """Helper to verify that augmentation flags work as expected."""
    logger.info("--- Validating Augmentations ---")
    
    cfg_no_aug = copy.deepcopy(cfg)
    cfg_no_aug.TRAIN['CHROMATIC'] = False
    cfg_no_aug.TRAIN['ADD_NOISE'] = False
    cfg_no_aug.TRAIN['SYN_CROP'] = False
    dm_no_aug = get_datamodule(dataset_name, data_path, batch_size=1, num_workers=0, config=cfg_no_aug)
    dm_no_aug.setup()
    original_batch = next(iter(dm_no_aug.train_dataloader()))
    assert original_batch is not None, "Failed to load batch for augmentation test."
    original_image = original_batch["image_color"]

    cfg_with_aug = copy.deepcopy(cfg)
    cfg_with_aug.TRAIN['CHROMATIC'] = True
    cfg_with_aug.TRAIN['ADD_NOISE'] = True
    cfg_with_aug.TRAIN['SYN_CROP'] = True
    dm_with_aug = get_datamodule(dataset_name, data_path, batch_size=1, num_workers=0, config=cfg_with_aug)
    dm_with_aug.setup()
    augmented_batch = next(iter(dm_with_aug.train_dataloader()))
    assert augmented_batch is not None, "Failed to load augmented batch."
    augmented_image = augmented_batch["image_color"]

    assert not torch.equal(original_image, augmented_image), "Chromatic/Noise augmentation FAILED."
    logger.info("Chromatic/Noise augmentation validation PASSED.")
    
    expected_size = cfg_with_aug.TRAIN['SYN_CROP_SIZE']
    assert augmented_image.shape[2] == expected_size and augmented_image.shape[3] == expected_size, \
        f"SYN_CROP validation FAILED: Expected {expected_size}, got {augmented_image.shape[2:]}."
    logger.info(f"SYN_CROP validation PASSED.")

@pytest.fixture(scope="module")
def metric_test_data():
    """Provides common data for metric tests."""
    y_true = np.array([
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 0]
    ])
    y_pred_perfect = np.copy(y_true)
    y_pred_partial = np.array([
        [1, 1, 1],
        [1, 0, 0],
        [0, 0, 0]
    ])
    y_pred_no_overlap = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 0]
    ])
    y_pred_empty = np.zeros_like(y_true)
    
    return {
        "true": y_true,
        "perfect": y_pred_perfect,
        "partial": y_pred_partial,
        "no_overlap": y_pred_no_overlap,
        "empty": y_pred_empty,
        "all_true": np.ones_like(y_true)
    }

# ---------------------------------------------------
# Test Suite for Metrics
# ---------------------------------------------------

def test_precision(metric_test_data):
    """Tests the precision metric with various scenarios."""
    data = metric_test_data
    assert precision(data["true"], data["perfect"]) == pytest.approx(1.0)
    # tp=3, fp=1 -> precision = 3/4 = 0.75
    assert precision(data["true"], data["partial"]) == pytest.approx(0.75)
    assert precision(data["true"], data["no_overlap"]) == pytest.approx(0.0)
    assert precision(data["true"], data["empty"]) == pytest.approx(0.0)
    # Test case where prediction is all 1s
    # tp=4, fp=5 -> precision = 4/9
    assert precision(data["true"], data["all_true"]) == pytest.approx(4/9)

def test_recall(metric_test_data):
    """Tests the recall metric with various scenarios."""
    data = metric_test_data
    assert recall(data["true"], data["perfect"]) == pytest.approx(1.0)
    # tp=3, fn=1 -> recall = 3/4 = 0.75
    assert recall(data["true"], data["partial"]) == pytest.approx(0.75)
    assert recall(data["true"], data["no_overlap"]) == pytest.approx(0.0)
    assert recall(data["true"], data["empty"]) == pytest.approx(0.0)
    # tp=4, fn=0 -> recall = 4/4 = 1.0
    assert recall(data["true"], data["all_true"]) == pytest.approx(1.0)

def test_f1_score(metric_test_data):
    """Tests the F1-score metric."""
    data = metric_test_data
    assert f1_score(data["true"], data["perfect"]) == pytest.approx(1.0)
    # p=0.75, r=0.75 -> f1 = 2 * (0.75 * 0.75) / (0.75 + 0.75) = 0.75
    assert f1_score(data["true"], data["partial"]) == pytest.approx(0.75)
    assert f1_score(data["true"], data["no_overlap"]) == pytest.approx(0.0)
    # p=4/9, r=1.0 -> f1 = 2 * (4/9 * 1) / (4/9 + 1) = (8/9) / (13/9) = 8/13
    assert f1_score(data["true"], data["all_true"]) == pytest.approx(8/13)

def test_intersection_over_union(metric_test_data):
    """Tests the IoU metric."""
    data = metric_test_data
    assert intersection_over_union(data["true"], data["perfect"]) == pytest.approx(1.0)
    # intersection=3, union=5 -> iou = 3/5 = 0.6
    assert intersection_over_union(data["true"], data["partial"]) == pytest.approx(0.6)
    assert intersection_over_union(data["true"], data["no_overlap"]) == pytest.approx(0.0)
    # intersection=4, union=9 -> iou = 4/9
    assert intersection_over_union(data["true"], data["all_true"]) == pytest.approx(4/9)

def test_iou_threshold(metric_test_data):
    """Tests the IoU thresholding function."""
    data = metric_test_data
    assert iou_threshold(data["true"], data["perfect"], threshold=0.9) is True
    assert iou_threshold(data["true"], data["partial"], threshold=0.5) is True
    assert iou_threshold(data["true"], data["partial"], threshold=0.7) is False
    assert iou_threshold(data["true"], data["no_overlap"], threshold=0.1) is False

def test_compute_metrics(metric_test_data):
    """Tests the main metric computation utility."""
    data = metric_test_data
    metrics_to_compute = ["precision", "iou", "iou_at_0.7"]
    results = compute_metrics(data["true"], data["partial"], metrics_to_compute)

    assert "precision" in results
    assert "iou" in results
    assert "iou_at_0.7" in results
    assert results["precision"] == pytest.approx(0.75)
    assert results["iou"] == pytest.approx(0.6)
    assert results["iou_at_0.7"] is False

    # Test for case-insensitivity and unknown metrics
    with pytest.raises(ValueError, match="not recognized"):
        compute_metrics(data["true"], data["partial"], ["Precision", "unknown_metric"])

def test_get_available_metrics():
    """Ensures the list of available metrics is correct."""
    available = get_available_metrics()
    expected = ['precision', 'recall', 'f1_score', 'iou', 'iou_at_0.7']
    assert isinstance(available, list)
    assert sorted(available) == sorted(expected)
