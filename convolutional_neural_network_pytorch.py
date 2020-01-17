# Imports
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Loading training set and test set
training_set = datasets.MNIST('./data',
                              train=True,
                              transform=transforms.ToTensor(),
                              download=True)

test_set = datasets.MNIST('./data',
                          train=False,
                          transform=transforms.ToTensor(),
                          download=True)

# Setting hyperparameters
learning_rate = 0.01
batch_size = 100
iterations = 2000  # Achieves 92+ test accuracy
epochs = int(iterations / (len(training_set) / batch_size))

# Making train and test loaders (making dataset iterable)
training_loader = torch.utils.data.DataLoader(dataset=training_set,
                                              batch_size=batch_size,
                                              shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                          batch_size=batch_size,
                                          shuffle=False)


# Creating convolutional neural network model class
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()

        # Convolution 1: Input 28x28x1, Filter 9x9x32 (same pad), Output 28x28x32
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=9, stride=1, padding=4, bias=True)
        self.relu1 = nn.ReLU();
        self.maxp1 = nn.MaxPool2d(2, 2)  # Output 14x14x32

        # Convolution 2: Input 14x14x32, Filter 7x7x64 (same pad), Output 14x14x64
        self.conv2 = nn.Conv2d(32, 64, 7, 1, 3, bias=True)
        self.relu2 = nn.ReLU();
        self.maxp2 = nn.MaxPool2d(2, 2)  # Output 7x7x64

        # Convolution 3: Input 7x7x64, Filter 3x3x128 (same pad), Output 7x7x128
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2, bias=True)
        self.relu3 = nn.ReLU();
        self.maxp3 = nn.MaxPool2d(2, 2)  # Output 3x3x128

        # Fully-connected layer 1: 768x512 (3x3x128 = 768)
        self.linear1 = nn.Linear(128 * 3 * 3, 512)
        self.relu4 = nn.ReLU()

        # Fully-connected layer 2: 512x10
        self.linear2 = nn.Linear(512, 10)

    def forward(self, x):
        # Conv1
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.maxp1(out)

        # Conv2
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.maxp2(out)

        # Conv3
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.maxp3(out)

        # Flattening the output
        out = out.view(-1, 3 * 3 * 128)

        # FC1
        out = self.linear1(out)
        out = self.relu4(out)

        # FC2
        out = self.linear2(out)

        return out


# Instantiating the model
model = ConvolutionalNeuralNetwork()

# Using cross-entropy loss
criterion = nn.CrossEntropyLoss()

# Using stochastic gradient descent as optimizer
optimizer = torch.optim.SGD(model.parameters(), learning_rate)

# Training loop
iter = 0
print('Now training...')
for epoch in range(epochs):
    for i, (images, labels) in enumerate(training_loader):

        # Convert data to Pytorch Variables
        images = Variable(images)
        labels = Variable(labels)

        # Clear (previous) gradients
        optimizer.zero_grad()

        # Forward pass through the model
        output = model(images)

        # Calculate loss
        loss = criterion(output, labels)

        # Calculate gradients from the loss
        loss.backward()

        # Update model's paramters
        optimizer.step()

        iter += 1

        if iter % 500 == 0:
            # Calculate Accuracy
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                images = Variable(images)

                # Forward pass to get output
                outputs = model(images)

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)

                # Total number of labels
                total += labels.size(0)

                # Total correct predictions
                correct += (predicted == labels).sum()

            accuracy = 100 * correct / total

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.data, accuracy))

# Testing the model
print('Testing model...')
for i, (images, labels) in enumerate(test_loader):
    images = Variable(images)

    # Forward pass to get output
    outputs = model(images)

    # Get predictions from the maximum value
    _, predicted = torch.max(outputs.data, 1)

    print('Prediction: ', end='')
    print(predicted[0], end=' ')
    print('Label for Image %d : ' % i, end='')
    print(labels[0])
