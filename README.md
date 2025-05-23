# Real-Time-Localized-Image-Enhancement-via-YOLOv8-and-Lightweight-Super-Resolution-Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of our real-time image enhancement framework that combines YOLOv8 segmentation with lightweight super-resolution. Achieves 3-20 FPS with 0.5 to 3 seconds latency on consumer GPUs(Nvidia 3060 Laptop GPU) by selectively enhancing only regions of interest.

## 📌 Key Features
- 🎯 YOLOv8-based segmentation for dynamic region selection
- 🔍 Lightweight ESRGAN for 2x super-resolution
- ⚡ CUDA optimization for <3s end-to-end latency

## 🛠️ Repository Structure
├── config.yaml # Configuration for data paths

├── mask.py # Segmentation mask transformation

├── train_gpu.py # Model training script with GPU

├── predict_output_gpu.py # predict mask area based on trained model with GPU

├── mainx2.py # complete pipeline for model carry out

├── requirements.txt # Python dependencies

└── LICENSE # MIT License


## 🚀 Quick Start
1. Install dependencies:
  - pip install -r requirements.txt

2. Train custom model:
  - python train.py --config config.yaml

3. Run inference on sample images:
  - python predict.py --input samples/ --config config.yaml

4. Run mainx2.py and mainx4.py:
  - python mainx2.py to run a complete version of video enhancement deployment with 2 times the pixels
  - python mainx2.py to run a complete version of video enhancement deployment with 4 times the pixels

5. (OPTIONAL) Customizing YOLO segmentation model
  - To customize a yolo model, prepare a dataset with the original, label it using the label tool
  - Generate a yolo file using mask.py to convert label pictures to a file that yolo can read
    
## 📊 Evaluation Metrics
- Metric	Target Performance
- Enhancement FPS:	3 - 20 (1920p*1080p input) on Nvidia 3060 Laptop GPU
- End-to-end Latency: 0.5 - 3 seconds

