import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import random_split, DataLoader
from collections import Counter

import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(torch.cuda.is_available())
print(DEVICE)


### Hyperparameters ###
TRAIN_TEST_SPLIT = 0.8
LR = 1.5e-4
EPOCHS = 15
BATCH_SIZE = 20

LR_FEATURE = 0.5e-5 
LR_CLASSIFIER = 1.5e-4 


### Data wrapping ###
transform = transforms.Compose([
    transforms.Resize((204, 75)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

DATA_DIR = "./imgs/data_loader"
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
train_size = int(len(dataset) * TRAIN_TEST_SPLIT)
test_size = len(dataset) - train_size

torch.manual_seed(42)
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


### Class-weighted Loss + Criterion ###
class_counts = Counter([label for _, label in dataset.samples])
total_samples = len(dataset)

class_weights = [total_samples / (len(class_counts) * count) for count in class_counts.values()]
class_weights = torch.tensor(class_weights).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)


def test(model, criterion, test_loader):
    print("# ### TEST")
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    misclassified_images = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += (preds[i] == label).item()
                class_total[label] += 1

                if preds[i] != labels[i]:
                    misclassified_images.append((images[i].cpu(), labels[i].item(), preds[i].item()))

    print(f"Test Loss: {test_loss / len(test_loader):.4f}")
    print(f"Accuracy: {correct / total * 100:.2f}%")
    print('-' * 25)
    for i, cls in enumerate(dataset.classes):
            print(f"Class {cls}: {class_correct[i] / class_total[i] * 100:.2f}% ({class_correct[i]}/{class_total[i]})")
        
    for idx, (image, true_label, predicted_label) in enumerate(misclassified_images):
        print(f"Image {idx + 1}: True Label = {dataset.classes[true_label]}, Predicted Label = {dataset.classes[predicted_label]}")

        #"""
        image_np = image.permute(1, 2, 0).numpy()
        image_np = image_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        image_np = np.clip(image_np, 0, 1)

        plt.imshow(image_np)
        plt.title(f"True: {dataset.classes[true_label]}, Predicted: {dataset.classes[predicted_label]}")
        plt.axis("off")
        plt.savefig(f"misclassified_{idx + 1}.png")
        #"""
        

model = models.efficientnet_v2_s(weights=None)

num_classes = 4 #len(train_dataset.classes)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

model.load_state_dict(torch.load("models/model7/model7.pth"))

test(model, criterion, test_loader)
