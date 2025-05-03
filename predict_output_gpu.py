import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def main():
    # set path
    model_path = r'C:\Users\Huihao Xing\Desktop\Graduate Spring\DS 542 Deep Learning\Project\segmentation\runs\segment\train9\weights\best.pt'
    image_path = r'C:\Users\Huihao Xing\Desktop\Graduate Spring\DS 542 Deep Learning\Project\segmentation\data\images\train\2007_000480.jpg'
    output_path = 'C:/Users/Huihao Xing/Desktop/Graduate Spring/DS 542 Deep Learning/Project/segmentation/runs/segment/train7/portrait_output.png'

    # check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"PyTorch CUDA avail: {torch.cuda.is_available()}")
    print(f"avil device: {device}")

    # transfer to GPU
    model = YOLO(model_path).to(device)
    print(f"model run on: {next(model.model.parameters()).device}")

    # read image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"can't load image: {image_path}")
    H, W = img.shape[:2]

    # GPU inference
    with torch.no_grad():
        results = model(img, verbose=False)

    # result processing
    for result in results:
        if result.masks is None:
            print("no subject detected")
            return

        # mask creating
        best_mask_idx = torch.argmax(result.boxes.conf).item()
        mask = result.masks.data[best_mask_idx]

        mask_np = mask.cpu().numpy()
        mask_resized = cv2.resize(mask_np, (W, H))
        binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255

        cv2.imwrite('binary_mask.png', binary_mask)

        # create background
        rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = binary_mask

        # save result
        cv2.imwrite(output_path, rgba)
        print(f"image mask save to: {output_path}")


        cv2.imshow('Portrait', rgba)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()