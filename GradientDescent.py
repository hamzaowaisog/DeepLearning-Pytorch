# Manual Predictions
# Manual Gradients Computation
# Manual Loss computation
# Manual Parameter updates

import numpy as np

# Linear regression
# f = w * x
x = np.array([1,2,3,4] , dtype = np.float32)
y = np.array([2,4,6,8] , dtype=np.float32)

w = 0.0 

#model prediction
def forward(x):
    return w*x

#loss = MSE
def loss (y, y_predicted):
    return ((y_predicted - y)**2).mean()

#gradient
#MSE = 1/N * (W*x - y) **2
#dJ/dw = 1/N 2x (w*x - y)

def gradient (x,y,y_predicted):
    return np.dot(2*x, y_predicted - y).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')
# Training
learning_rate = 0.01
n_iters = 15
# for i in range(n_iters):
for epochs in range(n_iters):
    # prediction = forward pass
    y_pred = forward(x)
    
    # loss
    l = loss(y, y_pred)
    
    # gradients
    dw = gradient(x, y, y_pred)
    
    # update weights
    w -= learning_rate * dw
    
    if epochs % 1 == 0:
        print(f'epoch {epochs+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')

# Manual Predictions
# Automatic Gradients Computation
# Manual Loss computation
# Manual Parameter updates

import torch

# Linear regression
# f = w * x
X = torch.tensor([1,2,3,4] , dtype = torch.float32)
Y = torch.tensor([2,4,6,8] , dtype=torch.float32)
W = torch.tensor(0.0 , dtype=torch.float32 , requires_grad=True)
# model prediction
def forward(x):
    return W*x

#loss = MSE
def loss (y, y_predicted):
    return ((y_predicted - y)**2).mean()

print(f'Prediction before training: f(5) = {forward(5).item():.3f}')
# Training
learning_rate = 0.01
n_iters = 30
# for i in range(n_iters):
for epochs in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)
    
    # loss
    l = loss(Y, y_pred)
    
    # gradients = backward pass
    l.backward() # dl/dw
    
    # update weights
    with torch.no_grad():
        W -= learning_rate * W.grad
    
    # zero gradients
    W.grad.zero_()
    
    if epochs % 1 == 0:
        print(f'epoch {epochs+1}: w = {W.item():.3f}, loss = {l.item():.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')