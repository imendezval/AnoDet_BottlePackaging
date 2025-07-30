"""
Real-Time Anomaly Detection from RTSP Camera Feed using EfficientNetV2-S

This script captures a live video stream from an RTSP camera, preprocesses frames, 
and uses a trained EfficientNetV2-S model to classify anomalies in the scene.

Features:
- Loads a pre-trained EfficientNetV2-S model for inference.
- Captures and processes frames from an RTSP camera stream.
- Applies region-of-interest (ROI) cropping and masking to each frame.
- Performs real-time inference and displays predictions on the video feed.
- Implements an error detection mechanism based on consecutive anomaly predictions.

Usage:
- The script continuously processes frames and displays the prediction.
- Press 'q' to exit the live video feed.
"""
import os
import cv2
import argparse
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models

class Config:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Real-Time Anomaly Detection from RTSP using EfficientNetV2-S")

        parser.add_argument("--model-path", type=str, required=True, help="Path to trained EfficientNetV2 model (.pth)")
        parser.add_argument("--rtsp-url", type=str, default="rtsp://192.168.60.101:8554/", help="RTSP camera URL")
        parser.add_argument("--mask-path", type=str, default="imgs/bin_mask_opt.jpg", help="Path to mask image")
        parser.add_argument("--num-classes", type=int, default=4, help="Number of output classes")
        parser.add_argument("--input-size", type=int, default=224, help="Input size for model (square)")
        self.args = parser.parse_args()

class LiveAnomalyDetector:
    def __init__(self, config: Config):
        self.cfg = config.args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = self._load_model()
        self.transform = self._build_transform()
        self.mask_resized = self._load_mask()

        # Frame + ROI Settings
        self.original_width = 1280
        self.original_height = 720
        self.new_width = 640
        self.new_height = 368
        self.original_roi = [525, 215, 150, 400]  # x, y, w, h
        self.scaled_roi = [
            int(self.original_roi[0] * (self.new_width / self.original_width)),
            int(self.original_roi[1] * (self.new_height / self.original_height)),
            int(self.original_roi[2] * (self.new_width / self.original_width)),
            int(self.original_roi[3] * (self.new_height / self.original_height))
        ]

    def _load_model(self):
        model = models.efficientnet_v2_s(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.cfg.num_classes)
        model.load_state_dict(torch.load(self.cfg.model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def _build_transform(self):
        return transforms.Compose([
            transforms.Resize((self.cfg.input_size, self.cfg.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _load_mask(self):
        mask = cv2.imread(self.cfg.mask_path)
        return cv2.resize(mask, (self.scaled_roi[2], self.scaled_roi[3]))

    def preprocess_frame(self, frame):
        A, B = 109, 262
        img_cropped = frame[A:A+204, B:B+75]
        masked_img = cv2.bitwise_and(img_cropped, self.mask_resized)
        masked_rgb = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)

        def resize_to_height(image, height):
            aspect_ratio = image.shape[1] / image.shape[0]
            width = int(aspect_ratio * height)
            return cv2.resize(image, (width, height))

        img_crop_resized = resize_to_height(img_cropped, self.new_height)
        masked_resized = resize_to_height(masked_img, self.new_height)
        concat = cv2.hconcat([img_crop_resized, masked_resized])

        pil_img = transforms.functional.to_pil_image(masked_rgb)
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        return input_tensor, concat

    def predict(self, input_tensor):
        with torch.no_grad():
            output = self.model(input_tensor)
        return output.argmax(dim=1).item(), output

    def run(self):
        print(cv2.getBuildInformation())
        camera = cv2.VideoCapture(self.cfg.rtsp_url, cv2.CAP_FFMPEG)
        if not camera.isOpened():
            print("Error: Could not open RTSP stream.")
            return

        prev_pred = 2
        counter = 0
        timer = 0

        while True:
            ret, frame = camera.read()
            if not ret:
                print("Failed to capture frame.")
                break

            input_tensor, concat_display = self.preprocess_frame(frame)
            pred_class, prediction = self.predict(input_tensor)

            # Anomaly stabilization logic
            if pred_class == prev_pred and pred_class != 2:
                counter += 1
            else:
                counter = 0

            # Prediction to label and color
            label_map = {
                0: ("Fallen After", (0, 0, 255)),
                1: ("Fallen Before", (0, 0, 255)),
                2: ("No Anomaly", (0, 255, 0)),
                3: ("No Lid", (0, 0, 255))
            }
            label_text, color = label_map[pred_class]
            cv2.putText(frame, f"Prediction: {label_text}", (10, 350), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=color, thickness=2)

            if counter >= 5 or (0 < timer <= 18):
                cv2.putText(frame, "ERROR!", (50, 190), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2, color=(0, 0, 255), thickness=3)
                timer += 1
            if timer == 20:
                timer = 0
                counter = 0

            prev_pred = pred_class
            print("Prediction:", prediction)

            # Display live video
            cv2.imshow("Live Feed", frame)
            cv2.imshow("Crop + Masked", concat_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        camera.release()
        cv2.destroyAllWindows()

def main():
    config = Config()
    detector = LiveAnomalyDetector(config)
    detector.run()

if __name__ == "__main__":
    main()