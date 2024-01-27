import torch
import numpy

x = torch.randn(3, requires_grad=True)
print(x)
z = x*x*2
#z = y.mean()
print(z)
v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
z.backward(v)  # dz/dx
print(x.grad)

# x.requires_grad(false)
# n = x.detach()
# with torch.no_grad():


# >>>>>>>>>>>>>>>>>>> Example <<<<<<<<<<<<<<<<<<<<<<<<<
print(">>>>>>>>>>>>>>>>>>> Example <<<<<<<<<<<<<<<<<<<<<<<<<")
weights = torch.ones(4, requires_grad=True)

for epoch in range(5):
    model_output = (weights * 3).sum()
    model_output.backward()
    print(weights.grad)
    # El gradiente se va acumulando
    weights.grad.zero_()  # Se deben vaciar los gradientes
