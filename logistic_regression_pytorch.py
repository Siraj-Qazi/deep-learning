# Imports
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils as utils

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
iterations = 3000
epochs = int(iterations / (len(training_set) / batch_size))

# Making train and test loaders (making dataset iterable)
training_loader = torch.utils.data.DataLoader(dataset=training_set,
                                              batch_size=batch_size,
                                              shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                          batch_size=batch_size,
                                          shuffle=False)


# Creating logistic regression model class
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out


input_size = 28 * 28  # MNIST input
output_size = 10  # Softmax output

# Instantiating the model
model = LogisticRegressionModel(input_size, output_size)

# Using cross-entropy loss for logistic regression
criterion = nn.CrossEntropyLoss()

# Using stochastic gradient descent as optimizer
optimizer = torch.optim.SGD(model.parameters(), learning_rate)

# Training loop
iter = 0
for epoch in range(epochs):
    for i, (images, labels) in enumerate(training_loader):

        # Convert data to Pytorch Variables
        images = Variable(images.view(-1, 28 * 28))
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
                images = Variable(images.view(-1, 28 * 28))

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
