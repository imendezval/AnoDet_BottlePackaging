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

"""
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
"""

### Class-weighted Loss + Criterion ###
class_counts = Counter([label for _, label in dataset.samples])
total_samples = len(dataset)

class_weights = [total_samples / (len(class_counts) * count) for count in class_counts.values()]
class_weights = torch.tensor(class_weights).to(DEVICE)


criterion = nn.CrossEntropyLoss(weight=class_weights)
"""

### Layer (Un)Freezing ###
for param in model.features.parameters():
    param.requires_grad = False

for param in model.classifier.parameters():
    param.requires_grad = True


unfreeze_schedule = {0: 2}#, 12: 4, 25: len(list(model.features.children()))}

for name, param in model.named_parameters():
    print(f"{name:30} {param.requires_grad}")
"""

### Scheduler ###
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
"""
for param in model.features[-5:].parameters():
    param.requires_grad = True
"""

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


        #train_accuracy = correct_train / total_train
        #train_accuracies.append(train_accuracy)

        model.eval()
        train_accuracy = evaluate_accuracy(model, train_loader)
        train_accuracies.append(train_accuracy)
        test_accuracy = evaluate_accuracy(model, test_loader)
        test_accuracies.append(test_accuracy)
        model.train()


        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        if epoch % 5 == 0:  # Save every 10 epochs
            torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pth")
            print(f"Checkpoint saved at epoch {epoch}")
            # Plot training loss
            plt.figure()
            plt.plot(range(1, len(train_losses) + 1), train_losses)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.savefig(f'Loss_6_checkpoint_epoch_{epoch}.png')
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
            plt.savefig(f'Accuracy_6_checkpoint_epoch_{epoch}.png')
            plt.clf()

    torch.save(model.state_dict(), "model6.pth")
    print(train_losses)

    # Plot training loss
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('Loss_6.png')
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
    plt.savefig('Accuracy_6.png')
    plt.clf()


def evaluate_accuracy(model, data_loader):
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
        

        
#train(model, criterion, optimizer, EPOCHS, train_loader, test_loader, unfreeze_schedule)

#"""
model = models.efficientnet_v2_s(weights=None)

num_classes = 4 #len(train_dataset.classes)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

model.load_state_dict(torch.load("checkpoint_epoch_5.pth"))

test(model, criterion, test_loader)
#"""







"""
remove
nd 1813
nd 1810
nd 2833
nd 3736
nd 3911
nd 4320
nd 4617
nd 4968
nd 5307
nd 5518
nd 5863
nd 6076
nd 6324
nd 6651
"""