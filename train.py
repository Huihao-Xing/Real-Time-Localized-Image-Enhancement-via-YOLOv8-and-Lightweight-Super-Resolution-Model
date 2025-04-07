from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel
import torch

# Add SegmentationModel to safe globals
torch.serialization.add_safe_globals([SegmentationModel])

model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)

model.train(data='config.yaml', epochs=1, imgsz=640)