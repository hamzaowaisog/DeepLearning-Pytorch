# 1) Design Model (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training Loop
#     - forward pass: compute prediction
#     - backward pass: gradients
#     - update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

## 0) Prepare data

x_numpy , y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

x = torch.from_numpy(x_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)

n_samples , n_features = x.shape

## 1)model
input_size = n_features
output_size = 1

model = nn.Linear(input_size,output_size)


## 2)Loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

## Move model and data to GPU

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
x = x.to(device)
y = y.to(device)

## 3)Training loop
num_epochs = 300

for epoch in range(num_epochs):
    #forward pass and loss
    y_predicted = model(x)
    loss = criterion(y_predicted, y)

    #backward pass
    loss.backward()

    #update
    optimizer.step()

    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item()}')
        
#plot
predicted = model(x).detach().cpu().numpy()
plt.plot(x_numpy, y_numpy, 'ro')
plt.plot(x_numpy, predicted, 'b')
plt.show()
