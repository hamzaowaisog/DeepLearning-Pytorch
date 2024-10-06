import torch
import numpy as np

x = torch.empty(3,2,3)
print(x)

y = torch.rand(3,2,3 , dtype=torch.float32)
print(y)
print(y.size())

#basic operations
z = torch.rand(2,2)
a = torch.rand(2,2)

print(z)
print(a)

print(torch.add(z,a))

#slicing tensors
b = torch.rand(5,3)
print(b)
print(b[1,1].item())

# reshaping tensors

c = torch.rand(4,4)
print(c)
print(c.view(2,8))

# convert numpy into tensors and vice versa
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

a = np.ones(5)
b = torch.from_numpy(a)
print(a)
print(b)

a += 1
print(a)
print(b)

#tensors using gpu
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = y.to(device)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))