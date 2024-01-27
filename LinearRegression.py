# 1) Design model (input, output size, foward pass)
# 2) Construct loss and optimizer
# 3) Training loop:
#   - Foward pass: compute prediction
#   - backward pass: gradients
#   - update weights

import torch
import torch.nn as nn
from sklearn import datasets
import numpy
import matplotlib.pyplot as plt

# 0) Prepare data
X_numpy, y_numpy = datasets.make_regression(n_samples=100,n_features=1,noise=50,random_state=123)

X = torch.from_numpy(X_numpy.astype(numpy.float32))
y = torch.from_numpy(y_numpy.astype(numpy.float32))
y = y.view(-1,1) # Super importante, definir las dimensiones de los tensores correctamente
n_samples,n_features = X.shape

# 1) Model
input_features = n_features
output_features = 1
linearRegression = nn.Linear(in_features=input_features,out_features=output_features)

# 2) Construct loss and optimizer
learning_rate = 1.0
loss = nn.MSELoss()
optimizer = torch.optim.Adam(linearRegression.parameters(),lr=learning_rate)

# 3) Training loop
epochs = 100
for epoch in range(epochs):
    # Foward pass: compute prediction
    y_predicted = linearRegression(X)
    
    # backward pass: gradients
    l = loss(y,y_predicted)
    l.backward()
    
    # update weights
    optimizer.step()
    optimizer.zero_grad()
    
    if (epoch % 10) == 0:
        print(f'Epoch {epoch + 1}: loss = {l:.4f}')
        
predictions = linearRegression(X).detach().numpy()
plt.scatter(X_numpy,y_numpy)
plt.plot(X_numpy,predictions,c='red') 
plt.savefig('LinearRegression_Torch.png')






