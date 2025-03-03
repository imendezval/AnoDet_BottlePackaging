import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DATA_DIR = "./imgs/data_loader"

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=transform)

BATCH_SIZE = 20
LR = 2e-4
EPOCHS = 50

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

input_channels = 3
input_size = 128
num_classes = len(train_dataset.classes)

class MyCNNModel(nn.Module):
    def __init__(self,  input_channels=input_channels, input_size = input_size, output_size=num_classes, conv_layers_config=None,
                 hidden_layers=[100, 100], activation=nn.ReLU, dropout_coef=0.0, use_dropout=False):

        super(MyCNNModel, self).__init__()

        if conv_layers_config is None:
          # Default configuration for the convolutional layers
          conv_layers_config = [{'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                                {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1}]

        # Initialize convolutional layer list
        conv_layers = []

        in_channels = input_channels

        # Appending convolutions to the list with desired parameters
        for convolution in conv_layers_config:
          conv_layers.append(nn.Conv2d(in_channels, convolution['out_channels'],
                                       kernel_size=convolution['kernel_size'],
                                       stride=convolution['stride'],
                                       padding=convolution.get('padding', 0)))
          conv_layers.append(activation())
          conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
          if use_dropout:
            conv_layers.append(nn.Dropout(dropout_coef))

          # Update input channels for the next layer using output channels from the previous
          in_channels = convolution['out_channels']

        # Convolutional layers into Sequential for feed forward
        self.conv_network = nn.Sequential(*conv_layers)

        # Flattened dimension of last convolution required to create fully connected layers (height*width*channels)
        # A function to calculate the height and width of the last convolution is therefore necessary
        conv_output_size = self.get_conv_output_size(input_size, conv_layers_config)
        conv_flattened_size = conv_output_size * conv_output_size * conv_layers_config[-1]['out_channels']

        # Initialize fully connected layer list
        fc_layers = []

        # Input to first hidden layer
        fc_layers.append(nn.Linear(conv_flattened_size, hidden_layers[0]))
        fc_layers.append(activation())
        if use_dropout:
          fc_layers.append(nn.Dropout(dropout_coef))

        # Fully connected hidden layers
        for layer_index in range(1, len(hidden_layers)):
          fc_layers.append(nn.Linear(hidden_layers[layer_index-1], hidden_layers[layer_index]))
          fc_layers.append(activation())
          if use_dropout:
            fc_layers.append(nn.Dropout(dropout_coef))

        # Last fully connected hidden layer to output
        fc_layers.append(nn.Linear(hidden_layers[-1], output_size))
        fc_layers.append(nn.Softmax(dim=1))

        self.fc_network = nn.Sequential(*fc_layers)


    def get_conv_output_size(self, conv_input_size, conv_layers_config):
      # Function to calculate output size of last convolution
      for convolution in conv_layers_config:
        kernel_size = convolution['kernel_size']
        stride = convolution['stride']
        padding = convolution.get('padding', 0)

        conv_input_size = (conv_input_size - kernel_size + 2 + padding) // stride
        conv_input_size = conv_input_size // 2

      return conv_input_size

    def forward(self, x):
      # Feed forward function
      x = self.conv_network(x)

      # Flatten the convolution for fully connected
      x = x.view(x.size(0), -1)

      x = self.fc_network(x)

      return x

def train(model, crit, opt, epochs, train_loader) -> None:
    print("# ### TRAIN")
    """function to perform the training process, using the defined model, criterion
    and optimizer objects. Prints train loss evolution at specific epoch increments
    and plots at the end of the loss throughout every epoch

    :param model: ML model instance extending nn.Module class
    :param crit: criterion, nn.*Loss() instance
    :param opt: optimizer, nn.optim.* instance
    :param train_loader: torch DataLoader instance containing training samples
    :return: None"""
    train_losses = []

    model.to(DEVICE)

    model.train()
    for epoch in range(epochs):
        losses = 0
        for images, labels in train_loader:
            opt.zero_grad()
            y_hat = model(images.to(DEVICE))
            loss = crit(y_hat, labels.to(DEVICE))

            # TODO: make the necessary update for the loss and optimizer
            loss.backward()
            opt.step()

            losses += loss.item()

        train_loss = losses / len(train_loader)
        train_losses.append(train_loss)
        if epoch != 0 and epoch % 5 == 0:
            print('Epoch: [{}/{}] - train loss: {}'.format(epoch, epochs, train_loss))

    # plot the process of training, visualizing losses
    plt.plot(range(0, len(train_losses)), train_losses)
    plt.show()


def test(model, crit, test_loader) -> None:
    print("# ### TEST")
    """function to perform the evaluation process, using the defined model and
    criterion. Prints test loss, overall accuracy and class-wise accuracy for
    samples of the test dataset.

    :param model: ML model instance extending nn.Module class
    :param crit: criterion, nn.*Loss() instance
    :param test_loader: torch DataLoader instance containing testing samples
    :return: None"""
    test_loss = 0

    # dictionaries used to evaluate for each class the accuracy of the model
    total_predictions = [0 for i in range(0, 10)]
    correct_predictions = [0 for i in range(0, 10)]

    model.to(DEVICE)

    model.eval()
    with torch.no_grad():
      for images, labels in test_loader:
          y_hat = model(images)
          _, pred = torch.max(y_hat, 1)
          loss = criterion(y_hat, labels)

          test_loss += loss.item()

          for idx, gt in enumerate(labels):
              # TODO: increment the total_predictions list at the index of the ground truth label
              total_predictions[gt] += 1
              if pred[idx] == gt:
                  correct_predictions[gt] += 1


    print('-' * 25)
    print('Loss:', test_loss / len(test_loader))
    print('Accuracy: {:.2f}%'.format(sum(correct_predictions) / sum(total_predictions) * 100))
    print('-' * 25)
    for i in range(0, 10):
        print('Label {}: {:.2f}% acc - {} presences'.format(i, correct_predictions[i] / total_predictions[i] * 100, total_predictions[i]))


conv_layers_config = [{'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                      {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                      {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1}]


CNNModel = MyCNNModel(input_channels=1, input_size = 28, output_size=10, conv_layers_config=conv_layers_config,
                      hidden_layers=[100, 50], activation=nn.ReLU, dropout_coef=0.2, use_dropout=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(CNNModel.parameters(), lr=LR)

train(
    model=CNNModel,
    crit=criterion,
    opt=optimizer,
    epochs=EPOCHS,
    train_loader=train_loader
)
test(
    model=CNNModel,
    crit=criterion,
    test_loader=test_loader
)