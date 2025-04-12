import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def main():
    # 配置路径（使用原始字符串避免转义问题）
    model_path = r'C:\Users\Huihao Xing\Desktop\Graduate Spring\DS 542 Deep Learning\Project\segmentation\runs\segment\train9\weights\best.pt'
    image_path = r'C:\Users\Huihao Xing\Desktop\Graduate Spring\DS 542 Deep Learning\Project\segmentation\data\images\train\2007_000480.jpg'
    output_path = 'C:/Users/Huihao Xing/Desktop/Graduate Spring/DS 542 Deep Learning/Project/segmentation/runs/segment/train7/portrait_output.png'

    # 检查GPU可用性
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"PyTorch CUDA可用: {torch.cuda.is_available()}")
    print(f"使用设备: {device}")

    # 加载模型并转移到GPU
    model = YOLO(model_path).to(device)
    print(f"模型运行在: {next(model.model.parameters()).device}")

    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}")
    H, W = img.shape[:2]

    # GPU推理
    with torch.no_grad():  # 禁用梯度计算以提升性能
        results = model(img, verbose=False)  # 禁用冗余输出

    # 处理结果
    for result in results:
        if result.masks is None:
            print("未检测到人物区域")
            return

        # 获取最佳掩码（选择置信度最高的检测）
        best_mask_idx = torch.argmax(result.boxes.conf).item()
        mask = result.masks.data[best_mask_idx]

        # 转换掩码到CPU并处理
        mask_np = mask.cpu().numpy()
        mask_resized = cv2.resize(mask_np, (W, H))
        binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255  # 二值化

        # 保存原始掩码（可选）
        cv2.imwrite('binary_mask.png', binary_mask)

        # 创建透明背景的人物图像
        rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = binary_mask  # 设置alpha通道

        # 保存结果（PNG格式保留透明背景）
        cv2.imwrite(output_path, rgba)
        print(f"人物抠图已保存至: {output_path}")

        # 可选：显示结果（需要GUI支持）
        cv2.imshow('Portrait', rgba)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()