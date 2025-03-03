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

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import random_split, DataLoader
from collections import Counter

import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Checking for GPU
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


### Model ###
model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
num_classes = 4 #len(train_dataset.classes)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
#print(model)
model = model.to(DEVICE)


### Variable LR per different layers + Optimizer ###
feature_params = list(model.features.parameters())
classifier_params = list(model.classifier.parameters())

param_groups = [
    {"params": feature_params, "lr": LR_FEATURE},
    {"params": classifier_params, "lr": LR_CLASSIFIER},
]


#optimizer = torch.optim.Adam(model.parameters(), lr=LR)
#optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

### Class-weighted Loss + Criterion ###
class_counts = Counter([label for _, label in dataset.samples])
total_samples = len(dataset)

class_weights = [total_samples / (len(class_counts) * count) for count in class_counts.values()]
class_weights = torch.tensor(class_weights).to(DEVICE)


criterion = nn.CrossEntropyLoss(weight=class_weights)

### Layer (Un)Freezing ###
for param in model.features.parameters():
    param.requires_grad = False

for param in model.classifier.parameters():
    param.requires_grad = True


unfreeze_schedule = {0: 2} #, 12: 4, 25: len(list(model.features.children()))}

for name, param in model.named_parameters():
    print(f"{name:30} {param.requires_grad}")


### Scheduler ###
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


def unfreeze_layers(model, num_layers_to_unfreeze):
    """
    Unfreezes the last `num_layers_to_unfreeze` layers of the feature extractor.
    """
    layers = list(model.features.children())
    for layer in layers[-num_layers_to_unfreeze:]:
        for param in layer.parameters():
            param.requires_grad = True

    #for name, param in model.named_parameters():
    #    print(f"{name:30} {param.requires_grad}")
    

def train(model, criterion, optimizer, epochs, train_loader, test_loader, unfreeze_schedule):
    """
    Trains the model with the given parameters.
    
    Parameters:
    - model: The neural network model
    - criterion: Loss function
    - optimizer: Optimizer for training
    - epochs: Number of training epochs
    - train_loader: DataLoader for training data
    - test_loader: DataLoader for test data
    - unfreeze_schedule: Dictionary specifying which layers to unfreeze at which epoch
    
    Saves model checkpoints and plots loss and accuracy at certain epochs.
    """
    print("# ### TRAIN")
    model.train()
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        # Gradually unfreeze layers
        if epoch in unfreeze_schedule:
            num_layers_to_unfreeze = unfreeze_schedule[epoch]
            print(f"Epoch {epoch + 1}: Unfreezing {num_layers_to_unfreeze} layers.")
            unfreeze_layers(model, num_layers_to_unfreeze)
        
        total_loss = 0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        model.eval()
        train_accuracy = evaluate_accuracy(model, train_loader)
        # train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)
        test_accuracy = evaluate_accuracy(model, test_loader)
        test_accuracies.append(test_accuracy)
        model.train()

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        # Save checkpoints every 5 epochs
        if epoch % 5 == 0:  
            torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pth")
            print(f"Checkpoint saved at epoch {epoch}")
            # Plot training loss
            plt.figure()
            plt.plot(range(1, len(train_losses) + 1), train_losses)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.savefig(f'Loss_8_checkpoint_epoch_{epoch}.png')
            plt.clf()

            # Plot training and test accuracy
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label="Train Accuracy", color="blue")
            plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label="Test Accuracy", color="orange")
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.title('Training + Test Accuracy per Epoch')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'Accuracy_8_checkpoint_epoch_{epoch}.png')
            plt.clf()

    torch.save(model.state_dict(), "model8.pth")
    print(train_losses)

    # Plot training loss
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('Loss_8.png')
    plt.clf()

    # Plot training and test accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label="Train Accuracy", color="blue")
    plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label="Test Accuracy", color="orange")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training + Test Accuracy per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('Accuracy_8.png')
    plt.clf()


def evaluate_accuracy(model, data_loader):
    """
    Evaluates the accuracy of the model on a given dataset.
    
    Parameters:
    - model: The neural network model
    - data_loader: DataLoader for evaluation
    
    Returns:
    - accuracy: Float value representing accuracy on the dataset
    """
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


train(model, criterion, optimizer, EPOCHS, train_loader, test_loader, unfreeze_schedule)