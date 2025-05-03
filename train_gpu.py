from ultralytics import YOLO
import torch


def main():
    # check pug
    print(f"PyTorch CUDA avail: {torch.cuda.is_available()}")

    # load model
    model = YOLO('yolov8n-seg.pt').to('cuda')
    print(f"model device: {next(model.model.parameters()).device}")

    # parameters setting
    model.train(
        data='config.yaml',
        epochs= 10,
        imgsz=640,
        workers=1,
        device='cuda'
    )


if __name__ == '__main__':
    main()  