import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out


x_train = np.array([i for i in range(11)], dtype=np.float32)
x_train = x_train.reshape(-1, 1)
y_train = np.array([2 * i + 1 for i in range(11)], dtype=np.float32)
y_train = y_train.reshape(-1, 1)

input_dim = 1
output_dim = 1

model = LinearRegressionModel(input_dim, output_dim)
criterion = nn.MSELoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 1000
for epoch in range(epochs):
    input_values = Variable(torch.from_numpy(x_train))
    labels = Variable(torch.from_numpy(y_train))

    optimizer.zero_grad()

    outputs = model(input_values)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    print('Epoch: {0} Loss: {1}'.format(epoch + 1, loss))

predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
print(predicted)

