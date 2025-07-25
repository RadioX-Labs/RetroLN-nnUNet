# RetroLN-nnUNet
Detecting Retroperitoneal Lymph Nodes using nnUNet

---

## Overview

**RetroLN-nnUNet** is an advanced deep learning pipeline for automated detection and segmentation of retroperitoneal lymph nodes in CT scans. Built on nnUNet v2 framework, this project introduces innovative **vertebral landmark-guided region extraction** for consistent anatomical alignment across diverse patient populations.

### Clinical Impact
- **Automated lymph node detection** in retroperitoneal region
- **Consistent anatomical alignment** using L2 vertebra as reference
- **Production-ready pipeline** with comprehensive validation
- **Research-grade accuracy** with clinical deployment potential

---

## Key Features

### **Anatomical Intelligence**
- **Vertebral Landmark Guidance**: Uses L2 vertebra for consistent ROI extraction
- **Multi-Vertebral Segmentation**: T12-L5 vertebrae identification via TotalSegmentator
- **Anatomical Coordinate System**: Automatic AP/SI/LR axis determination

### **Deep Learning Excellence**
- **nnUNet v2 Framework**: State-of-the-art medical image segmentation
- **3D U-Net Architecture**: 6-stage encoder-decoder with skip connections
- **Deep Supervision**: Multi-resolution loss for enhanced training
- **Mixed Precision Training**: Optimized for Tesla V100 GPUs

### **Production Ready**
- **Parallel Processing**: Multi-core CPU and GPU acceleration
- **Robust Error Handling**: Comprehensive validation and logging
- **Memory Optimization**: Efficient processing of large 3D volumes
- **Format Flexibility**: Support for NIfTI formats (.nii, .nii.gz)

---

## Architecture

### **Network Specifications**

| Component | Specification |
|-----------|---------------|
| **Architecture** | PlainConvUNet (3D) |
| **Input Dimensions** | [1, 64, 192, 160] |
| **Patch Size** | [64, 192, 160] |
| **Encoder Stages** | 6 |
| **Feature Maps** | [32, 64, 128, 256, 320, 320] |
| **Kernel Sizes** | [[1,3,3], [3,3,3], [3,3,3], [3,3,3], [3,3,3], [3,3,3]] |
| **Strides** | [[1,1,1], [1,2,2], [2,2,2], [2,2,2], [2,2,2], [2,2,2]] |

### **Processing Pipeline**
<div align="center">
    <img src="https://github.com/RadioX-Labs/RetroLN-nnUNet/blob/main/resources/nnunetv2_RLN_pipeline.jpg" alt="Segmentation Results" width="800"/>
</div>


### **U-Net Layer Structure**

---

## Dataset

### **Dataset Specifications**

| Property | Value |
|----------|-------|
| **Dataset ID** | Dataset600 |
| **Institution** | PGIMER Chandigarh |
| **Total Cases** | 834 |
| **Modality** | CT |
| **Classes** | 2 (Background, Lymph Node) |
| **Format** | NIfTI (.nii.gz) |

### **Image Properties**

| Aspect | Original | Resampled |
|--------|----------|-----------|
| **Spacing (mm)** | [2.5, 0.79, 0.79] | [2.5, 0.79, 0.79] |
| **Shape** | [58, 185, 154] | [71, 227, 188] |
| **Intensity Range** | HU values | [0.0, 1.0] |

### **Intensity Statistics**
- **Mean**: 0.767 ± 0.029
- **Range**: [0.0, 1.0] (normalized)
- **Percentiles**: 0.5% = 0.681, 99.5% = 0.849

---

## Setup

### **Prerequisites**
- Python 3.8+
- CUDA 11.2+ (for GPU acceleration)
- 16GB+ RAM recommended
- 50GB+ free disk space

### **One-Line Installation (Recommended)**
```bash
git clone https://github.com/RadioX-Labs/RetroLN-nnUNet.git && cd RetroLN-nnUNet && pip install -r requirements.txt
```

### **Conda Environment**

```bash
# Create and activate environment
conda create -n retroln python=3.8
conda activate retroln

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install nnUNet
pip install nnunetv2

# Install additional dependencies
pip install -r requirements.txt
```

### **Quick Processing**
```bash
# 1. Extract vertebral landmarks
python vert_seg.py --input_dir ./data/ct --output_dir ./vertebrae --num_processes 4

# 2. Crop retroperitoneal region
python crop.py --ct_dir ./data/ct --vert_seg_dir ./vertebrae --output_dir ./processed

# 3. Run nnUNet training
nnUNetv2_train Dataset600 3d_fullres 0
```

---

## Usage

### **Complete Pipeline**

#### **Step 1: Vertebral Segmentation**
```bash
python vert_seg.py \
    --input_dir /path/to/ct/scans \
    --output_dir /path/to/vertebrae \
    --num_processes 4 \
    --batch_size 10
```

**Parameters:**
- `--input_dir`: Directory containing CT scans (*_0000.nii.gz)
- `--output_dir`: Output directory for vertebral segmentations
- `--num_processes`: Number of parallel processes (default: 2)
- `--batch_size`: Number of images to process (0 = all)
- `--fast`: Enable fast mode (lower quality, faster processing)

#### **Step 2: ROI Cropping**
```bash
python crop.py \
    --ct_dir /path/to/ct/scans \
    --mask_dir /path/to/masks \
    --vert_seg_dir /path/to/vertebrae \
    --output_dir /path/to/processed \
    --parallel \
    --num_processes 4
```

**Parameters:**
- `--ct_dir`: Directory containing CT volumes
- `--mask_dir`: Directory containing segmentation masks (optional)
- `--vert_seg_dir`: Directory containing vertebral segmentations
- `--output_dir`: Output directory for processed volumes
- `--parallel`: Enable parallel processing
- `--no_normalize`: Disable CT intensity normalization

#### **Step 3: nnUNet Training**
```bash
# Set environment variables
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"

# Plan and preprocess
nnUNetv2_plan_and_preprocess -d 600

# Train model
nnUNetv2_train 600 3d_fullres 0 --npz
```

### **Single Case Processing**

```bash
python crop.py \
    --single \
    --ct_file /path/to/case.nii.gz \
    --mask_file /path/to/mask.nii.gz \
    --vert_seg_file /path/to/vertebrae/case \
    --ct_output /path/to/output/ct.nii.gz \
    --mask_output /path/to/output/mask.nii.gz
```

---

## Configuration

### **Training Configuration**

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Batch Size** | 2 | Limited by GPU memory |
| **Patch Size** | [64, 192, 160] | 3D patch dimensions |
| **Initial LR** | 0.01 | Starting learning rate |
| **Momentum** | 0.99 | SGD momentum |
| **Weight Decay** | 3e-05 | L2 regularization |
| **Max Epochs** | 1000 | Maximum training epochs |
| **Deep Supervision** | True | Multi-resolution loss |

### **Hardware Requirements**

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | 8GB VRAM | Tesla V100 32GB |
| **RAM** | 16GB | 32GB+ |
| **Storage** | 100GB | 500GB SSD |
| **CPU** | 4 cores | 8+ cores |

---

## Results

### **Performance Metrics**

### **Processing Performance**

### **Validation Results**

---

## Technical Details

### **Loss Function**
- **Combined Loss**: Dice + Cross-Entropy
- **Dice Component**: Memory-efficient soft Dice loss
- **CE Component**: Robust cross-entropy loss
- **Deep Supervision**: Multi-scale loss at each decoder level

### **Data Augmentation**
- **Spatial**: Random rotations, scaling, elastic deformation
- **Intensity**: Gaussian noise, brightness, contrast adjustment
- **Anatomical**: Vertebral landmark-aware transformations

### **Anatomical Cropping Parameters**
- **AP Dimension**: 180mm crop with 30mm posterior offset from L2
- **LR Dimension**: 150mm centered on vertebral centerline
- **SI Dimension**: Full vertebral extent (T12-L5)
- **Reference Point**: L2 vertebra anterior margin

### **Quality Assurance**
- **Input Validation**: NIFTI format and dimension checks
- **Anatomical Validation**: Vertebral landmark verification
- **Output Validation**: Segmentation quality metrics
- **Error Recovery**: Automatic fallback strategies

---

## License

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http.creativecommons.org/licenses/by-nc-sa/4.0/).

### **Citation**

If you use this work in your research, please cite:


---

## Acknowledgments

### **Clinical Collaboration**
- **PGIMER Chandigarh** - Dataset and clinical expertise
- **Medical Imaging Team** - Annotation and validation

### **Technical Foundation**
- **nnUNet Team** - Framework and methodology
- **TotalSegmentator** - Vertebral segmentation tool
- **PyTorch Community** - Deep learning framework

---

<div align="center">

### **Star this repository if it helps you!**

[![GitHub stars](https://img.shields.io/github/stars/radiox-labs/RetroLN-nnUNet.svg?style=social&label=Star)](https://github.com/radiox-labs/RetroLN-nnUNet)
[![GitHub forks](https://img.shields.io/github/forks/radiox-labs/RetroLN-nnUNet.svg?style=social&label=Fork)](https://github.com/radiox-labs/RetroLN-nnUNet/fork)

[⬆ Back to Top](#-retroln-nnunet)

</div>
