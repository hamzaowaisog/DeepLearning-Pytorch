import torch
import torch.nn as nn
import numpy as np

def softmax(x):
    return np.exp(x)/ np.sum(np.exp(x),axis=0)

x = np.array([2.0,1.0,0.1])
outputs = softmax(x)
print('softmax numpy:', outputs)

x = torch.tensor([2.0,1.0,0.1])
outputs = torch.softmax(x , dim=0)
print('Softmax torch:' ,outputs )

## cross entropy

def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss

Y = np.array([1,0,0])
Y_pred = np.array([0.7,0.2,0.1])
loss = cross_entropy(Y,Y_pred)
print('Loss numpy:', loss)


loss = nn.CrossEntropyLoss()
# nsamples
y = torch.tensor([2,0,1])
# nsamples x nclasses = 3x3 
y_pred = torch.tensor([[2.0, 1.0, 0.1], [1.0, 2.0, 0.1], [0.1, 0.2, 0.7]])
l = loss(y_pred, y)
print('Loss torch:', l.item())


_, predictions = torch.max(y_pred,1)
print(predictions)