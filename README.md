# MRI_Segmentation_with_U-Net_MobileNetV2_encoder
Lightweight U-Net with MobileNetV2 encoder (via segmentation-models-pytorch) for 2D MRI/CT segmentation on Kaggle’s “MRI dataset for detection and analysis.” Converts 3D NIfTI to 2D, z-score normalizes images, binarizes masks (nearest-neighbor). Trains with BCE+Dice, AMP, grad clipping, early stopping; runs a threshold sweep and outputs visual tools overlay predictions.


# The pipeline:
- Loads 3D NIfTI volumes (.nii) with NiBabel
- Converts them into 2D axial slices
- Normalizes images and binarizes masks (with nearest-neighbor resizing)
- Trains a MobileNetV2-U-Net with BCE + Dice loss, AMP, and gradient clipping
- Evaluates with Dice/IoU/Precision/Recall/F1
- Sweeps thresholds to pick the best inference cutoff
- Visualizes predictions with overlays (and optional largest-CC post-processing for presentation)


# Acknowledgements
Dataset: MRI dataset for detection and analysis (Kaggle).
Kaggle: https://www.kaggle.com/datasets/sudipde25/mri-dataset-for-detection-and-analysis/data

Model: segmentation-models-pytorch (U-Net + MobileNetV2).
https://github.com/Rumit95/Semantic-Segmentation-of-Image 


# Trained on
kaggle accelerator==TPU VM v3-8

# Prerequisites
## Core DL
torch>=1.10
torchvision>=0.11

## Segmentation Models (exact pins used in notebook)
segmentation-models-pytorch==0.3.3
timm==0.9.2
pretrainedmodels==0.7.4
efficientnet-pytorch==0.7.1

## Data & vision utils
albumentations>=1.3.0,<2.0.0
opencv-python>=4.7.0,<5.0.0
nibabel>=5.1.0

## Metrics / plotting / tools
scikit-learn>=1.0
matplotlib>=3.6
numpy>=1.23
pandas>=1.5
tqdm>=4.65
scipy>=1.9
seaborn>=0.12
