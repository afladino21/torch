import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
input_size = 28 * 28
hidden_size = 100
num_clasess = 10
num_epochs = 3
batch_size = 64
learning_rate = 1.0e-3

# MNIST
train_dataset = torchvision.datasets.MNIST(
    root='data/MNIST', train=True, transform=transforms.ToTensor(), download=False)

test_dataset = torchvision.datasets.MNIST(
    root='data/MNIST', train=False, transform=transforms.ToTensor(), download=False)

# DataLoader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size)

# Model


class NeuronalNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_clasess) -> None:
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_clasess)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


model = NeuronalNetwork(input_size=input_size,
                        hidden_size=hidden_size, num_clasess=num_clasess)

#loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
# trainin loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # ? No entiendo muy bien esta parte
        images = images.view(-1, input_size).to(device)
        labels = labels.to(device)

        # Forward
        outs = model.forward(images) # O simplemente pasarle el modelo model(images)
        loss = criterion(outs, labels)

        # Backward
        loss.backward()

        # update weights
        optimizer.step()
        optimizer.zero_grad()

        if (i+1) % 10 == 0:
            print(
                f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, Loss: {loss.item()}')

n_correct = 0
n_samples = 0
for images, labels in test_loader:
    images = images.view(-1, input_size).to(device)
    labels = labels.to(device)
    outs = model.feedforward(images).detach()
    # value,index
    _, predictions = torch.max(outs, dim=1)
    n_samples += labels.shape[0]
    n_correct += (predictions == labels).sum().item()

acc = 100.0 * n_correct / n_samples
print(f'Accuracy: {acc}')
