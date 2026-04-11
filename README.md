# Multi-Task Perception Pipeline (DA6401 - Assignment 2)

**Report Link:** [Weights & Biases Report](https://api.wandb.ai/links/me22b101-indian-institute-of-technology-madras/3enyrytt)

---

## Overview

This repository contains a unified, multi-task Convolutional Neural Network (CNN) built from scratch using PyTorch. The model is trained on the **Oxford-IIIT Pet Dataset** and is designed to perform three distinct computer vision tasks simultaneously through a shared feature extraction backbone:

1. **Classification:** Identifying the specific breed of the pet (37 classes).
2. **Object Detection / Localization:** Drawing a precise bounding box around the subject.
3. **Semantic Segmentation:** Generating a pixel-level trimap mask (Foreground, Background, Border).

## Key Features & Architectural Highlights

* **Custom VGG11 Backbone:** A shared encoder architecture modified to support dynamic Batch Normalization toggling.
* **U-Net Style Decoder:** A symmetric upsampling network with skip connections for high-fidelity segmentation masks.
* **Custom Dropout Implementation:** A mathematically verifiable, hand-written dropout layer to mitigate generalization gaps in the dense classification head.
* **Transfer Learning Analysis:** Codebase supports dynamic freezing strategies (`strict`, `partial`, and `full` fine-tuning) to evaluate task interference and convergence stability.
* **Deep Visualization Hooks:** Includes inference scripts utilizing PyTorch forward hooks to extract and visualize intermediate feature maps (low-level edge detectors vs. high-level semantic heatmaps).
* **Robust Metric Tracking:** Fully integrated with Weights & Biases for tracking CrossEntropy, Smooth L1 Loss, mIoU, Macro F1, and Macro Dice Scores.

## Repository Structure

```text
├── data/
│   └── pets_dataset.py       # Dataloaders, normalization, and augmentation
├── models/
│   ├── __init__.py
│   ├── vgg11.py              # Custom VGG11 Backbone & Custom Dropout
│   ├── classification.py     # Dense Classification Head
│   ├── localization.py       # Bounding Box Regression Head
│   ├── segmentation.py       # U-Net Decoder
│   └── multitask.py          # Unified Multi-Task Perception Model
├── checkpoints/              # Directory for saved model weights
├── test_images/              # "In-the-wild" images for final pipeline testing
├── train.py                  # Main training loop with W&B logging & TL strategies
├── inference.py              # Evaluation scripts (Feature extraction, overlays, metrics)
├── requirements.txt          # Python dependencies
└── README.md
