"""
EfficientNetV2-S Model Evaluation Script

This script evaluates a trained EfficientNetV2-S image classification model on a test dataset.
It computes overall and per class accuracy, test loss, and saves misclassified images for analysis.
It also:
- Supports class-weighted loss for imbalanced datasets
- Computes per class accuracy statistics
- Saves and logs misclassified test images with predicted vs. true labels
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Config:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Evaluate EfficientNetV2-S on a test set")

        parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model (.pth)")
        parser.add_argument("--data-dir", type=str, default="./imgs/data_loader", help="Path to dataset root directory")
        parser.add_argument("--batch-size", type=int, default=20, help="Batch size for test loader")
        parser.add_argument("--num-classes", type=int, default=4, help="Number of output classes")
        parser.add_argument("--save-dir", type=str, default="./misclassified", help="Directory to save misclassified images")

        self.args = parser.parse_args()

class EfficientNetTester:
    def __init__(self, config: Config):
        self.cfg = config.args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self._prepare_data()
        self._prepare_model()
        self._prepare_loss()

        os.makedirs(self.cfg.save_dir, exist_ok=True)

    def _prepare_data(self):
        transform = transforms.Compose([
            transforms.Resize((204, 75)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = datasets.ImageFolder(root=self.cfg.data_dir, transform=transform)
        self.test_loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=False)
        self.dataset = dataset

    def _prepare_model(self):
        self.model = models.efficientnet_v2_s(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, self.cfg.num_classes)
        self.model.load_state_dict(torch.load(self.cfg.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def _prepare_loss(self):
        class_counts = Counter([label for _, label in self.dataset.samples])
        total_samples = len(self.dataset)
        class_weights = [total_samples / (len(class_counts) * count) for count in class_counts.values()]
        class_weights = torch.tensor(class_weights).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

    def test(self):
        print("# ### TEST")
        test_loss = 0
        correct = 0
        total = 0
        class_correct = [0] * self.cfg.num_classes
        class_total = [0] * self.cfg.num_classes
        misclassified_images = []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                for i in range(len(labels)):
                    true_label = labels[i]
                    pred_label = preds[i]
                    class_correct[true_label] += (pred_label == true_label).item()
                    class_total[true_label] += 1

                    if pred_label != true_label:
                        misclassified_images.append((images[i].cpu(), true_label.item(), pred_label.item()))

        print(f"Test Loss: {test_loss / len(self.test_loader):.4f}")
        print(f"Overall Accuracy: {correct / total * 100:.2f}%")
        print('-' * 25)
        for i, cls_name in enumerate(self.dataset.classes):
            acc = class_correct[i] / class_total[i] * 100 if class_total[i] else 0.0
            print(f"Class {cls_name}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")

        self._save_misclassified(misclassified_images)

    def _save_misclassified(self, misclassified):
        for idx, (image, true_label, pred_label) in enumerate(misclassified):
            image_np = image.permute(1, 2, 0).numpy()
            image_np = image_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
            image_np = np.clip(image_np, 0, 1)

            plt.imshow(image_np)
            plt.title(f"True: {self.dataset.classes[true_label]} | Pred: {self.dataset.classes[pred_label]}")
            plt.axis("off")
            save_path = os.path.join(self.cfg.save_dir, f"misclassified_{idx + 1}.png")
            plt.savefig(save_path)
            plt.close()

            print(f"Saved misclassified image {idx + 1} to {save_path}")

def main():
    config = Config()
    tester = EfficientNetTester(config)
    tester.test()

if __name__ == "__main__":
    main()