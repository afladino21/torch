
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy
import math


class WineDataset(Dataset):
    def __init__(self, transform) -> None:
        self.transform = transform
        xy = numpy.loadtxt("data/wine/winequality-red.csv",
                           delimiter=",", dtype=numpy.float32, skiprows=1)

        self.x = xy[:, 1:]
        self.y = xy[:, 0].reshape(-1, 1)
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        sample = self.x[index, :], self.y[index, :]
        if self.transform:
            return self.transform(sample)

    def __len__(self):
        return self.n_samples


class ToTensor():
    def __call__(self, sample):
        inputs, target = sample
        return torch.from_numpy(inputs), torch.from_numpy(target)


class MulTransform():
    def __init__(self, factor) -> None:
        self.factor = factor

    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target


dataset = WineDataset(transform=ToTensor())
inputs, target = dataset[0]
print(inputs)
print(type(inputs), type(target))

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset = WineDataset(transform=composed)
inputs, target = dataset[0]
print(inputs)
print(type(inputs), type(target))
