import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import Trainer

# hyper parameters
input_size = 28 * 28
hidden_size = 100
num_clasess = 10
num_epochs = 3
batch_size = 64
learning_rate = 1.0e-3


class LitNeuronalNetwork(pl.LightningModule):
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

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        images, labels = batch
        images = images.view(-1, input_size)
        output = self(images)
        loss = F.cross_entropy(output, labels)
        tensorboard_logs = {'train_loss':loss}
        return {'loss':loss,'log':tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def train_dataloader(self):
        train_dataset=torchvision.datasets.MNIST(root='../data/MNIST',
        train=True, 
        transform=transforms.ToTensor(), 
        download=False)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, 
            batch_size=batch_size, 
            num_workers=3,
            shuffle=True)
        return train_loader 

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        images, labels = batch
        images = images.view(-1, input_size)
        output = self(images)
        loss = F.cross_entropy(output, labels)
        return {'val_loss':loss}

    def validation_dataloader(self):
        validation_dataset=torchvision.datasets.MNIST(root='../data/MNIST',
        train=False, 
        transform=transforms.ToTensor(), 
        download=False)
        validation_dataloader = torch.utils.data.DataLoader(
            dataset=validation_dataset, 
            batch_size=batch_size, 
            num_workers=3,
            shuffle=False)
        return validation_dataloader

if __name__ == '__main__':
    # fast_dev_run: Only run 1 batch in order to check if everything is ok
    trainer = Trainer(max_epochs=num_epochs,fast_dev_run=False) #max_epochs=num_epochs
    model = LitNeuronalNetwork(input_size=input_size,hidden_size=hidden_size,num_clasess=num_clasess)
    trainer.fit(model=model)
