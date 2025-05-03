import os
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

from ultralytics import YOLO
import cv2
import numpy as np
import torch
import mss
import time
import queue
import threading
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

class FastEnhancementPipeline:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {self.device.upper()}")

        # load YOLO model
        self.seg_model = YOLO('yolov8n-seg.pt', verbose=False).to(self.device).eval()

        # load RealESRGAN
        model_path = r'C:\Users\Huihao Xing\Desktop\Graduate Spring\DS 542 Deep Learning\Project\segmentation\RealESRGAN_x2plus.pth'
        sr_model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=2, num_feat=64, num_block=23, num_grow_ch=32)
        sr_checkpoint = torch.load(model_path, map_location=self.device)
        sr_model.load_state_dict(sr_checkpoint['params_ema'], strict=False)
        sr_model.to(self.device).eval()

        self.sr_model = RealESRGANer(
            scale=2, model_path=model_path, model=sr_model,
            tile=0,
            half=self.device == 'cuda', device=self.device
        )

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        resized_frame = cv2.resize(frame, (320, 240))

        with torch.no_grad():
            results = self.seg_model(resized_frame, imgsz=240, conf=0.5, verbose=False)[0]

        if results.masks is None:
            return frame

        masks = results.masks.data.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        person_masks = [masks[i] for i, cls in enumerate(classes) if int(cls) == 0]

        if len(person_masks) == 0:
            return frame

        combined_mask = np.clip(np.sum(person_masks, axis=0), 0, 1)
        combined_mask = cv2.resize(combined_mask, (w, h))

        alpha = (combined_mask > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        output = frame.copy()

        for cnt in contours:
            x, y, ww, hh = cv2.boundingRect(cnt)

            if ww * hh < 500:
                continue

            patch = frame[y:y+hh, x:x+ww]

            if max(patch.shape[:2]) > 200:
                patch = cv2.resize(patch, (patch.shape[1]//2, patch.shape[0]//2))

            try:
                enhanced_patch, _ = self.sr_model.enhance(patch, outscale=2)
                enhanced_patch = cv2.resize(enhanced_patch, (ww, hh), interpolation=cv2.INTER_CUBIC)
                output[y:y+hh, x:x+ww] = enhanced_patch
            except Exception as e:
                print(f"增强失败: {e}")

        return output

class ScreenCaptureThread(threading.Thread):
    def __init__(self, frame_queue, monitor):
        super().__init__()
        self.frame_queue = frame_queue
        self.monitor = monitor
        self.running = True

    def run(self):
        with mss.mss() as sct:
            while self.running:
                screen = np.array(sct.grab(self.monitor))
                frame = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                time.sleep(0.01)

    def stop(self):
        self.running = False

class FrameProcessingThread(threading.Thread):
    def __init__(self, frame_queue, output_queue, pipeline):
        super().__init__()
        self.frame_queue = frame_queue
        self.output_queue = output_queue
        self.pipeline = pipeline
        self.running = True
        self.process_mode = False

    def run(self):
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                if self.process_mode:
                    processed = self.pipeline.process_frame(frame)
                    self.output_queue.put((processed, True))
                else:
                    self.output_queue.put((frame, False))

    def stop(self):
        self.running = False

    def toggle_mode(self):
        self.process_mode = not self.process_mode
        print(f"enhance mode {'on' if self.process_mode else 'off'}")

def main():
    pipeline = FastEnhancementPipeline()

    with mss.mss() as temp_sct:
        monitor = temp_sct.monitors[1]

    capture_area = {
        "top": monitor["top"] ,
        "left": monitor["left"] ,
        "width": monitor["width"],
        "height": monitor["height"]
    }

    frame_queue = queue.Queue(maxsize=5)
    output_queue = queue.Queue(maxsize=5)

    capture_thread = ScreenCaptureThread(frame_queue, capture_area)
    processing_thread = FrameProcessingThread(frame_queue, output_queue, pipeline)

    capture_thread.start()
    processing_thread.start()

    cv2.namedWindow('enhance preview', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('enhance preview', 1920, 1080)
    cv2.moveWindow('enhance preview', 0, 0)
    cv2.setWindowProperty('enhance preview', cv2.WND_PROP_TOPMOST, 1)

    while True:
        if not output_queue.empty():
            display, enhanced = output_queue.get()

            if enhanced:
                cv2.putText(display, 'Enhance ON', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(display, 'Enhance OFF', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('enhance preview', display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('t'):
            processing_thread.toggle_mode()

    capture_thread.stop()
    processing_thread.stop()
    capture_thread.join()
    processing_thread.join()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
