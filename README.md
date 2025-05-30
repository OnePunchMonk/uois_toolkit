
# 🧠 **uois_toolkit**  
A toolkit for **Unseen Object Instance Segmentation (UOIS)**  

---

## 📚 **Index**
- [🧠 **uois\_toolkit**](#-uois_toolkit)
  - [📚 **Index**](#-index)
  - [🚀 **Setup**](#-setup)
  - [📦 **Datasets**](#-datasets)
  - [🧪 **Testing Locally**](#-testing-locally)
  - [📤 **PyPI Publishing Steps 💡**](#-pypi-publishing-steps-)
  - [💻 **Usage Example**](#-usage-example)

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
