import torch

a = torch.LongTensor([8, 2])
b = torch.LongTensor([8, 2])
print(a[b].shape)