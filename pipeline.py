# 1) Design model (input, output size, foward pass)
# 2) Construct loss and optimizer
# 3) Training loop:
#   - Foward pass: compute prediction
#   - backward pass: gradients
#   - update weights

import torch 
import torch.nn as nn   

class LinearRegresion(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        # Define layers
        self.lin = nn.Linear(input_dim,output_dim) 
    
    def forward(self,x):
        return self.lin(x)
        
X = torch.tensor([[1],[2],[3],[4]],dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]],dtype=torch.float32)

n_samples,n_features = X.shape
input_size = n_features
output_size = n_features
model = LinearRegresion(input_size,output_size)
x_test = torch.tensor([5],dtype=torch.float32)
print(f'Prediction before training: f(5): {model.forward(x_test).item():.3f}')
# Training 
loss = nn.MSELoss()
learning_rate = 1.0e-1
n_iters = 100

optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

for epoch in range(n_iters):
    # Prediction 
    y_pred = model.forward(X)
    
    # Loss
    l = loss(Y,y_pred)
    
    # Gradients = backward pass
    l.backward() # dl / dw
    
    # Update Weights
    optimizer.step()
    
    # Zero gradients
    optimizer.zero_grad()
    
    if epoch % 10 == 0:
        [w,b] = model.parameters()
        print(model.parameters())
        print(f'Epoch {epoch + 1}: w = {w[0][0].item()}, loss = {l:.8f}')

print(f'Prediction after training: f(5): {model.forward(x_test).item():.3f}')