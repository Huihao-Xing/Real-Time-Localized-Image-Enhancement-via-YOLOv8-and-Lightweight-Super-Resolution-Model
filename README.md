# Real-Time-Localized-Image-Enhancement-via-YOLOv8-and-Lightweight-Super-Resolution-Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of our real-time image enhancement framework that combines YOLOv8 segmentation with lightweight super-resolution. Achieves 30+ FPS on consumer GPUs by selectively enhancing only regions of interest.

## 📌 Key Features
- 🎯 YOLOv8-based segmentation for dynamic region selection
- 🔍 Lightweight ESRGAN for 2x-4x super-resolution
- ⚡ TensorRT optimization for <700ms end-to-end latency
- 📊 Quantitative evaluation (mAP, IoU, FPS benchmarks)

## 🛠️ Repository Structure
├── config.yaml # Configuration for model parameters
├── mask.py # Segmentation mask processing
├── predict.py # Inference pipeline
├── train.py # Model training script
├── requirements.txt # Python dependencies
└── LICENSE # MIT License


## 🚀 Quick Start
1. Install dependencies:
pip install -r requirements.txt
*Make Sure to run following commands:
pip install --upgrade ultralytics

2. Run inference on sample images:
python predict.py --input samples/ --config config.yaml

3. Train custom model:
python train.py --config config.yaml

📊 Evaluation Metrics
Metric	Target Performance
Segmentation mAP	≥0.75
Enhancement FPS	≥30 (1080p input)
End-to-end Latency	<700ms


Further information is under active construction... for current contact, please email huxing@bu.edu
