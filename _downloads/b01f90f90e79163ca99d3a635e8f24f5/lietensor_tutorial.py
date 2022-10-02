"""
LieTensor Tutorial
==================

"""

import torch
import pypose as pp


######################################################################
# 1. Intialization
# ---------------------------------------
# 

a = pp.so3(torch.randn(2,3))
x = pp.identity_SE3(2,1)
y = pp.randn_se3(2,2)
print('a:', a, '\nx.shape:', x.shape, '\nx.gshape:', x.lshape)
print(x.lview(2))
print(y)


######################################################################
# All arguments in PyTorch are supported
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

a = pp.randn_SO3(3, device="cuda:0", dtype=torch.double, requires_grad=True)
b = pp.identity_like(a, device="cpu")
a, b
t = a.float()
a, t


######################################################################
# 2. Slicing and Shaping
# ---------------------------------------
# 

A = pp.randn_SO3(2,2)
B = pp.randn_SO3(2,1)
C = torch.cat([A,B], dim=1)         # Tensor cat
C[0,1] = pp.randn_SO3(1)            # Slicing set
D = C[1,:].Log()                    # Slicing get
E, F = torch.split(C, [1,2], dim=1) # Tensor split
print('A:', A.lshape)
print('B:', B.lshape)
print('C:', C.lshape)
print('D:', D.lshape)
print('E:', E.lshape)
print('F:', F.lshape)


######################################################################
# 3. Basic Operations
# ---------------------------------------
# 

(x * y.Exp()).Inv().Log()


######################################################################
# 4. Adjoint Transforms
# ---------------------------------------
# 

X = pp.randn_Sim3(6, dtype=torch.double)
a = pp.randn_sim3(6, dtype=torch.double)
b = X.AdjT(a)
print((X * b.Exp() - a.Exp() * X).abs().mean() < 1e-7)

X = pp.randn_SE3(8)
a = pp.randn_se3(8)
b = X.Adj(a)
print((b.Exp() * X - X * a.Exp()).abs().mean() < 1e-7)


######################################################################
# 5. Grdients
# ---------------------------------------
# 

x = pp.randn_so3(3, sigma=0.1, requires_grad=True, device="cuda")
assert x.is_leaf
loss = (x.Exp().Log()**2).sin().sum() # Just test, No physical meaning
loss.backward()
y = x.detach()
loss, x.grad, x, y


######################################################################
# 6. Test a Module
# ---------------------------------------
# 

from torch import nn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class TestNet(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.weight = pp.Parameter(pp.randn_so3(n))

    def forward(self, x):
        return self.weight.Exp() * x


n,epoch = 4, 5
net = TestNet(n).cuda()

optimizer = torch.optim.SGD(net.parameters(), lr = 0.2, momentum=0.9)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4], gamma=0.5)

print("Before Optimization:\n", net.weight)
for i in range(epoch):
    optimizer.zero_grad()
    inputs = pp.randn_SO3(n).cuda()
    outputs = net(inputs)
    loss = outputs.abs().sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    print(loss)

print("Parameter:", count_parameters(net))
print("After Optimization:\n", net.weight)