import torch
import torch.nn as nn
import numpy
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
# 0: Prepare data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.from_numpy(X_train.astype(numpy.float32))
X_test = torch.from_numpy(X_test.astype(numpy.float32))
y_train = torch.from_numpy(y_train.astype(numpy.float32))
y_test = torch.from_numpy(y_test.astype(numpy.float32))

y_train = y_train.view(-1, 1)
y_test = y_test.view(-1, 1)


# 1: Model


class LogisticRegression(nn.Module):
    def __init__(self, input_dims) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dims, 1)

    def forward(self, x):
        prediction = torch.sigmoid(self.linear(x))
        return prediction


logit = LogisticRegression(n_features)

# 2: Loss & Optimizer
loss = nn.BCELoss()
optimizer = torch.optim.Adam(logit.parameters(), lr=1e-1)
epochs = 100

# 3: Training loop
for epoch in range(epochs):
    # Forward and loss
    y_pred = logit.forward(X_train)
    l = loss(y_pred, y_train)  # !Importante el orden

    # backward pass
    l.backward()

    # update weights
    optimizer.step()
    optimizer.zero_grad()

    if ((epoch+1) % 10) == 0:
        print(f'Epoch {epoch+1}, loss: {l.item():.4f}')

logit.eval()
y_pred = logit.forward(X_test).detach().round()
acc = y_pred.eq(y_test).sum() / y_test.shape[0]
print(f'acc test: {acc:.4f}')
