import torch
import numpy as np


data = [[1, 2], [3, -4]]

x_data = torch.tensor(data)

x_ones_copy = torch.ones_like(x_data) # retains shape of x populated with ones

print(x_data, "\n\n", x_ones_copy)


rand = torch.rand((2,3))

print(rand)
print(rand.shape)
print(rand.dtype)
