'''
epoch = 1 forward and backward pass of ALL training samples

batch_size = number of training samples in one forward & backward pass

number of iterations = number of passes, each pass using [batch size] number of samples

e.g. 100 samples, batch_size=20 --> 100/20 = 5 iterations for 1 epoch  
'''

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy
import math


class WineDataset(Dataset):
    def __init__(self) -> None:
        xy = numpy.loadtxt("data/wine/winequality-red.csv",
                           delimiter=",", dtype=numpy.float32, skiprows=1)

        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, 0].reshape(-1, 1))
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index, :], self.y[index, :]

    def __len__(self):
        return self.n_samples


dataset = WineDataset()
batch_size = 4
dataLoader = DataLoader(dataset=dataset, batch_size=batch_size,
                        shuffle=True, num_workers=2)

# Training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/batch_size)
print(total_samples)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataLoader):
        if (i+1) % 5 == 0:
            print(
                f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')

# torchvision.datesets.MNIST()
# fashion-mnist
