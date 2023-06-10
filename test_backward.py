import torch

def func(x):
    return torch.sum(x)

x = torch.ones(5) * 2
x.requires_grad_()

y = func(x)
print(y)

y.backward()

print(x.grad)