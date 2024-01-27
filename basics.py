import torch
import numpy 

x = torch.rand(2,3) # torch.empty(dims )
#torch.zeros(dims)
#torch.ones(dims)
x = torch.tensor([[2.3,5.5],[3.0,2.6]]) #dtype
y = torch.rand(2,2)
z = x @ y 
y.add_(x) #Inplace  #z = torch.add(x,y)
z = torch.mul(x,y)
z = torch.div(x,y)
t = torch.rand(5,5)
# print(t[1,1].item())

#? Reshape
m = torch.rand(4,4)
y = m.view(16,1) 

#*####################### convert torch to numpy ########################  
a = torch.ones(5)
#print(a)
b = a.numpy()
#print(b)
a.add_(1)
#print(a)
#print(b)
#*####################### convert numpy to torch ########################
an = numpy.ones((3,3),dtype=numpy.float32)
bn = torch.from_numpy(an)
#print(bn)
#*####################### move to GPU ########################
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5,device=device)
    y = torch.ones(5)
    y = y.to(device)
    z = x+y
    z = z.to("cpu")
    print(z.numpy())

x = torch.ones(5,5,requires_grad=True) 