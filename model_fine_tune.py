import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DATA_DIR = "./imgs/data_loader"
# Hyperparameters
LR = 2e-4
EPOCHS = 50
BATCH_SIZE = 20


# Data wrapping
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)


# Model
model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
num_classes = 3#len(train_dataset.classes)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
#print(model)
model = model.to(DEVICE)
"""
for param in model.features.parameters():
    param.requires_grad = False
for param in model.features[-5:].parameters():  # Unfreeze last 5 layers
    param.requires_grad = True
"""

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def train(model, criterion, optimizer, epochs, train_loader):
    print("# ### TRAIN")
    model.train()
    train_losses = []

    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    # Plot training loss
    plt.plot(range(1, len(train_losses) + 1), train_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()


def test(model, criterion, test_loader):
    print("# ### TEST")
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

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

    print(f"Test Loss: {test_loss / len(test_loader):.4f}")
    print(f"Accuracy: {correct / total * 100:.2f}%")
    print('-' * 25)
    for i, cls in enumerate(train_dataset.classes):
        print(f"Class {cls}: {class_correct[i] / class_total[i] * 100:.2f}% ({class_correct[i]}/{class_total[i]})")


train(model, criterion, optimizer, EPOCHS, train_loader)
test(model, criterion, test_loader)