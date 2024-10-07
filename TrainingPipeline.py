# 1) Design Model (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training Loop
#     - forward pass: compute prediction
#     - backward pass: gradients
#     - update weights


#Manual Prediction
#Automatic Gradient Computation
#Automatic Loss Computation
#Automatic Parameter updates
# f = w * x

import torch
import torch.nn as nn

# Linear regression
# f = w * x
x = torch.tensor([1,2,3,4] , dtype = torch.float32)
y = torch.tensor([2,4,6,8] , dtype=torch.float32)
w  = torch.tensor(0.0 , dtype=torch.float32 , requires_grad=True)

# model prediction
def forward(x):
    return w*x

print(f'Prediction before training: f(5) = {forward(5).item():.3f}')

# Training
learning_rate = 0.01
n_iters = 30

# loss = MSE
loss = nn.MSELoss()
optimizer = torch.optim.SGD([w], lr = learning_rate)

for epochs in range(n_iters):
    # prediction = forward pass
    y_pred = forward(x)
    
    # loss
    l = loss(y, y_pred)
    
    # gradients = backward pass
    l.backward() # dl/dw
    
    # update weights
    optimizer.step()
    
    # zero gradients
    optimizer.zero_grad()
    
    if epochs % 1 == 0:
        print(f'epoch {epochs+1}: w = {w:.3f}, loss = {l:.8f}')
print(f'Prediction after training: f(5) = {forward(5).item():.3f}')

#Automatic Prediction
#Automatic Gradient Computation
#Automatic Loss Computation
#Automatic Parameter updates
# f = w * x

x = torch.tensor([[1],[2],[3],[4]] , dtype = torch.float32)
y = torch.tensor([[2],[4],[6],[8]] , dtype=torch.float32)
n_samples , n_features = x.shape

x_test = torch.tensor([5] , dtype = torch.float32)


print(n_samples, n_features)

input_size = n_features
output_size = n_features
model = nn.Linear(input_size, output_size)

print(f'Prediction before training: f(5) = {model(x_test).item():.3f}')

# Training
learning_rate = 0.01
n_iters = 30
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

for epochs in range(n_iters):
    # prediction = forward pass
    y_pred = model(x)
    
    # loss
    l = loss(y, y_pred)
    
    # gradients = backward pass
    l.backward() # dl/dw
    
    # update weights
    optimizer.step()
    
    # zero gradients
    optimizer.zero_grad()
    
    if epochs % 1 == 0:
        [w,b] = model.parameters()
        print(f'epoch {epochs+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')
        
print(f'Prediction after training: f(5) = {model(x_test).item():.3f}')
