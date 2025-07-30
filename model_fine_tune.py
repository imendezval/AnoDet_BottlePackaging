"""
Image Classification Training Script using EfficientNetV2-S

This script trains an image classification model using EfficientNetV2-S with a 
custom classifier head. It includes:
- Data loading and transformation using PyTorch's `torchvision` module.
- Class-weighted loss function to handle class imbalances.
- Layer freezing and gradual unfreezing for transfer learning.
- Variable learning rates for different parts of the model.
- Training loop with accuracy evaluation and checkpoint saving.

Hyperparameters, dataset paths, and model settings can be customized.
"""
import argparse
import os
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def parse_unfreeze_schedule(schedule_str):
    try:
        return {int(k): int(v) for k, v in (pair.split(':') for pair in schedule_str.split(','))}
    except Exception:
        raise argparse.ArgumentTypeError("Unfreeze schedule must be in format like '0:2,12:4'")

class Config:
    def __init__(self):
        parser = argparse.ArgumentParser(description="EfficientNetV2-S Trainer")

        # Paths
        parser.add_argument("--data-dir", type=str, default="./imgs/data_loader")
        parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")

        # Training settings
        parser.add_argument("--num-classes", type=int, default=4)
        parser.add_argument("--epochs", type=int, default=15)
        parser.add_argument("--batch-size", type=int, default=20)
        parser.add_argument("--train-test-split", type=float, default=0.8)

        # Learning rates
        parser.add_argument("--lr-feature", type=float, default=5e-6)
        parser.add_argument("--lr-classifier", type=float, default=1.5e-4)

        # Layer unfreezing
        parser.add_argument("--unfreeze-schedule", type=parse_unfreeze_schedule, default={0: 2},
                            help="Epoch:Layers format (e.g., '0:2,12:4')")

        self.args = parser.parse_args()

class ImageClassifierTrainer:
    def __init__(self, config: Config):
        self.cfg = config.args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.train_loader, self.test_loader, self.dataset = self._prepare_data()
        self.model = self._build_model()
        self.optimizer = self._build_optimizer()
        self.criterion = self._build_criterion()

        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)

    def _prepare_data(self):
        transform = transforms.Compose([
            transforms.Resize((204, 75)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = datasets.ImageFolder(root=self.cfg.data_dir, transform=transform)
        train_size = int(len(dataset) * self.cfg.train_test_split)
        test_size = len(dataset) - train_size
        torch.manual_seed(42)
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.cfg.batch_size)

        return train_loader, test_loader, dataset

    def _build_model(self):
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.cfg.num_classes)
        model.to(self.device)

        for param in model.features.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

        return model

    def _build_optimizer(self):
        feature_params = list(self.model.features.parameters())
        classifier_params = list(self.model.classifier.parameters())

        param_groups = [
            {"params": feature_params, "lr": self.cfg.lr_feature},
            {"params": classifier_params, "lr": self.cfg.lr_classifier},
        ]
        return torch.optim.AdamW(param_groups, weight_decay=1e-4)

    def _build_criterion(self):
        class_counts = Counter([label for _, label in self.dataset.samples])
        total_samples = len(self.dataset)
        class_weights = [total_samples / (len(class_counts) * count) for count in class_counts.values()]
        return nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(self.device))

    def _unfreeze_layers(self, num_layers):
        layers = list(self.model.features.children())
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    def _evaluate_accuracy(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total

    def _plot_metrics(self, losses, train_accs, test_accs, epoch):
        plt.figure()
        plt.plot(range(1, len(losses) + 1), losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig(os.path.join(self.cfg.checkpoint_dir, f'Loss_epoch_{epoch}.png'))
        plt.close()

        plt.figure()
        plt.plot(range(1, len(train_accs) + 1), train_accs, label='Train Acc')
        plt.plot(range(1, len(test_accs) + 1), test_accs, label='Test Acc')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Test Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.cfg.checkpoint_dir, f'Accuracy_epoch_{epoch}.png'))
        plt.close()

    def train(self):
        print("# Starting Training")
        train_losses, train_accuracies, test_accuracies = [], [], []

        for epoch in range(self.cfg.epochs):
            if epoch in self.cfg.unfreeze_schedule:
                num_layers = self.cfg.unfreeze_schedule[epoch]
                self._unfreeze_layers(num_layers)
                print(f"Epoch {epoch+1}: Unfreezing {num_layers} layers")

            self.model.train()
            total_loss = 0

            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            train_losses.append(avg_loss)

            train_acc = self._evaluate_accuracy(self.train_loader)
            test_acc = self._evaluate_accuracy(self.test_loader)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)

            if epoch % 5 == 0 or epoch == self.cfg.epochs - 1:
                print(f"Epoch {epoch+1}/{self.cfg.epochs}, Loss: {avg_loss:.4f}")
                torch.save(self.model.state_dict(), os.path.join(self.cfg.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"))
                print(f"Checkpoint saved at epoch {epoch}")
                self._plot_metrics(train_losses, train_accuracies, test_accuracies, epoch)

        torch.save(self.model.state_dict(), os.path.join(self.cfg.checkpoint_dir, "final_model.pth"))
        print("Final model saved.")

def main():
    config = Config()
    trainer = ImageClassifierTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
