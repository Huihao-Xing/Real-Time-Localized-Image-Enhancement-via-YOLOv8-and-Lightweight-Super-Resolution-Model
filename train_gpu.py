from ultralytics import YOLO
import torch


def main():
    # 检查CUDA
    print(f"PyTorch CUDA可用: {torch.cuda.is_available()}")

    # 加载模型
    model = YOLO('yolov8n-seg.pt').to('cuda')
    print(f"模型设备: {next(model.model.parameters()).device}")  # 应输出cuda:0

    # 训练配置（关键修改：workers=0或1）
    model.train(
        data='config.yaml',
        epochs= 10,
        imgsz=640,
        workers=1,  # Windows必须设为0或1
        device='cuda'  # 显式指定
    )


if __name__ == '__main__':
    main()  # Windows多进程必须的保护