from ultralytics import YOLO

import cv2


model_path = 'C:/Users/Huihao Xing/Desktop/Graduate Spring/DS 542 Deep Learning/Project/segmentation/runs/segment/train/weights/best.pt'

image_path = 'C:/Users/Huihao Xing/Desktop/Graduate Spring/DS 542 Deep Learning/Project/segmentation/person_jpg/2007_002639.jpg'

img = cv2.imread(image_path)
H, W, _ = img.shape

model = YOLO(model_path)

results = model(img)

for result in results:
    if result.masks is not None:
        for j, mask in enumerate(result.masks.data):
            mask = mask.numpy() * 255
            mask = cv2.resize(mask, (W, H))
            cv2.imwrite('./output.png', mask)
    else:
        print("No masks detected for this result.")