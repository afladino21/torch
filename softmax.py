import torch
import torch.nn as nn
import numpy

# Softmax
x = torch.tensor([1.4, 2.0, 0.9])
out = torch.softmax(x, dim=0)
print(out)

# Cross entropy
#! In pytorch, the cross entropy loss applies softmax, so the input is the logits
#! The last layer, should not be softmax in pytorch
loss = nn.CrossEntropyLoss()
Y = torch.tensor([1])
Y_pred_good = torch.tensor([[0.1, 1.4, 0.8]])
Y_pred_bad = torch.tensor([[1.8, 1.0, 1.5]])
l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)
print(f'L1: {l1:.4f}')
print(f'L2: {l2:.4f}')
