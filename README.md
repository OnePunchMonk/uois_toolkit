# 🧠 **uois_toolkit**  
A toolkit for **Unseen Object Instance Segmentation (UOIS)**  

---
## Setup and Install

```bash
git clone https://github.com/OnePunchMonk/uois_toolkit.git
cd uois_toolkit
pip install -e .
```

## Testing

```bash
python -m pytest test/test_datamodule.py -v --dataset_path tabletop=C:\path\to\tabletop --dataset_path ocid=C:\path\to\OCID-dataset --dataset_path osd=C:\path\to\OSD --dataset_path robot_pushing=C:\path\to\pushing_data --dataset_path iteach_humanplay=C:\path\to\humanplay-data
```

OR

```bash
!pytest test/testmodule.py --dataset_path "ocid=./datasets/ocid"
```





## 📚 **Index**
- [🧠 **uois\_toolkit**](#-uois_toolkit)
  - [📚 **Index**](#-index)
  - [🚀 **Setup**](#-setup)
  - [📦 **Datasets**](#-datasets)
  - [🧪 **Testing Locally**](#-testing-locally)
  - [📤 **PyPI Publishing Steps 💡**](#-pypi-publishing-steps-)
  - [💻 **Usage Example**](#-usage-example)
  - [⚙️ How to Run the Module](#-how-to-run-the-module)

---

## 🚀 **Setup**
```bash
# Create a conda environment
conda create -n uois-toolkit python=3.10
conda activate uois-toolkit

# Set CUDA path (use your installed CUDA version)
export CUDA_HOME=/usr/local/cuda-12.6

# Install dependencies
pip install torch torchvision opencv-python numpy scipy pycocotools
pip install git+https://github.com/facebookresearch/detectron2@65184fc057d4fab080a98564f6b60fae0b94edc4
```

---

## 📦 **Datasets**
- Download TOD, OCID, OSD, RobotPushing, and iTeach-HumanPlay datasets from [Box](https://utdallas.box.com/v/uois-datasets).
- Copy `OSD-0.2/` to `OSD-0.2-depth/` folder.
- Place all data in the `DATA/` directory.

🔗 **iTeach-HumanPlay Dataset Links**:
- **D5**: [Download](https://utdallas.box.com/v/iTeach-HumanPlay-D5)  
- **D40**: [Download](https://utdallas.box.com/v/iTeach-HumanPlay-D40)  
- **Test**: [Download](https://utdallas.box.com/v/iTeach-HumanPlay-Test)

---

## 🧪 **Testing Locally**
```bash
rm -rf build dist *egg*  # Clean previous builds
pip install -e .         # Install in editable mode
```

---

## 📤 **PyPI Publishing Steps 💡**
<details>
<summary>Click to expand</summary>

```bash
# Install build tools
python -m pip install build twine

# Clean previous builds
rm -rf build/ dist/ *.egg-info

# Build the distribution
python -m build
python setup.py sdist bdist_wheel  # Ensure version is updated in setup.py

# Upload to PyPI (ensure you have your PyPI token configured)
twine upload dist/*
```

</details>

---

## 💻 **Usage Example**
```bash
python test/test_datasets.py
```

If everything is working, you should see logs like:
```
INFO:uois_toolkit.core.datasets.iteach_humanplay:2859 images for dataset iteach_humanplay_object_train
INFO:uois_toolkit.core.datasets.iteach_humanplay:11796 images for dataset iteach_humanplay_object_train
INFO:uois_toolkit.core.datasets.iteach_humanplay:902 images for dataset iteach_humanplay_object_test
INFO:uois_toolkit.core.datasets.ocid:2390 images for dataset ocid_object_train
INFO:uois_toolkit.core.datasets.ocid:2390 images for dataset ocid_object_test
INFO:uois_toolkit.core.datasets.tabletop:280000 images for dataset tabletop_object_train
INFO:uois_toolkit.core.datasets.tabletop:28000 images for dataset tabletop_object_test
INFO:uois_toolkit.core.datasets.robot_pushing:321 images for dataset robot_pushing_object_train
INFO:uois_toolkit.core.datasets.robot_pushing:107 images for dataset robot_pushing_object_test
INFO:uois_toolkit.core.datasets.osd:111 images for dataset osd_object_train
INFO:uois_toolkit.core.datasets.osd:111 images for dataset osd_object_test
```

---

## ⚙️ How to Run the Module

This toolkit is designed to be validated and used through its testing suite and can be easily integrated into your own projects.

### Prerequisites: Environment and Data

Before running the module, ensure you have:
1.  **Set up the environment**: Follow the instructions in the [**Setup**](#-setup) section to create a `conda` environment and install all dependencies.
2.  **Downloaded the datasets**: Make sure you have downloaded the required datasets as described in the [**Datasets**](#-datasets) section and have noted their paths.

### Method 1: Running the Full Test Suite (Recommended)

The most reliable way to ensure everything is configured correctly is to run the built-in `pytest` suite. This will validate the dataloaders, augmentations, and metric calculations for each dataset.

Open your terminal and run the following command, replacing `/path/to/your/data/...` with the actual paths to your datasets:

```bash
python -m pytest test/test_datamodule.py -v --log-cli-level=INFO \
  --dataset_path tabletop=/path/to/your/data/tabletop \
  --dataset_path ocid=/path/to/your/data/ocid \
  --dataset_path osd=/path/to/your/data/osd \
  --dataset_path robot_pushing=/path/to/your/data/robot_pushing \
  --dataset_path iteach_humanplay=/path/to/your/data/iteach_humanplay
```

**Command Breakdown:**
- `python -m pytest`: Runs pytest as a module to ensure it uses the correct environment.
- `test/test_datamodule.py`: Specifies the test file to run.
- `-v`: Enables verbose mode, which provides more detailed output.
- `--dataset_path <name>=<path>`: This is a crucial argument. You must provide the path for each dataset you wish to test.

### Method 2: Example Usage in a Python Script

You can also import and use the datamodule directly in your own code.

```python
from uois_toolkit import get_datamodule, cfg
import pytorch_lightning as pl

# 1. Specify the dataset name and path
dataset_name = "tabletop"
data_path = "/path/to/your/data/tabletop"

# 2. Get the datamodule
# You can customize the configuration by modifying the `cfg` object before this call
data_module = get_datamodule(
    dataset_name=dataset_name,
    data_path=data_path,
    batch_size=4,
    num_workers=2,
    config=cfg
)

# 3. You can now use this data_module with a PyTorch Lightning Trainer
# model = YourLightningModel()
# trainer = pl.Trainer(accelerator="auto")
# trainer.fit(model, datamodule=data_module)

# Or simply inspect a batch of data
data_module.setup()
train_loader = data_module.train_dataloader()
batch = next(iter(train_loader))

print(f"Successfully loaded a batch for the {dataset_name} dataset!")
print("Image tensor shape:", batch["image_color"].shape)
```
