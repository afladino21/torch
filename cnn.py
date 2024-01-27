import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hiper-párametros
num_epochs = 10
batch_size = 4
learning_rate = 1e-3
# Images transformation
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# Load CIFAR10 dataset
train_dataset = torchvision.datasets.CIFAR10(
    root='data/CIFAR10', train=True, download=False, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(
    root='data/CIFAR10', train=False, download=False, transform=transform)
# Data loader
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False)
classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


def plot_image(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(numpy.transpose(npimg, (1, 2, 0)))
    plt.show()


dataiter = iter(train_loader)
images, labels = next(dataiter)
plot: bool = False

print(images[0].shape[1], images[0].shape[2])

if plot:
    # show images
    plot_image(torchvision.utils.make_grid(images))

# Model implementation


class ConvNet(nn.Module):
    def __init__(self, batch_size, num_channels, W, H) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.W = W
        self.H = H
        self.conv1 = nn.Conv2d(in_channels=self.num_channels,
                               out_channels=32, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.calculate_dimensions()
        self.fc1 = nn.Linear(
            self.out_channels*self.out_height*self.out_width, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def calculate_dimensions(self):
        x = torch.randn(self.batch_size, self.num_channels, self.W, self.H)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        self.out_channels, self.out_height, self.out_width = x.size(
            1), x.size(2), x.size(3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, self.out_channels*self.out_height*self.out_width)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


cnn = ConvNet(batch_size=batch_size, num_channels=3,
              W=images[0].shape[1], H=images[0].shape[2])
criterion = nn.CrossEntropyLoss()
optimzer = torch.optim.SGD(params=cnn.parameters(), lr=learning_rate)
n_total_steps = len(train_loader)

# Training loop
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = cnn.forward(images)
        loss = criterion(outputs, labels)

        # backward pass
        loss.backward()
        optimzer.step()
        optimzer.zero_grad()

        if (i+1) % 10 == 0:
            print(
                f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, Loss: {loss.item()}')

print(">>>>>>>>>>>>>>>>> Finished training loop <<<<<<<<<<<<<<<<<<")

n_correct = 0
n_samples = 0
for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)
    outs = cnn.forward(images).detach()
    # value,index
    _, predictions = torch.max(outs, dim=1)
    n_samples += labels.shape[0]
    n_correct += (predictions == labels).sum().item()

acc = 100.0 * n_correct / n_samples
print(f'Accuracy: {acc}')
