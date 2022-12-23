"""
LieTensor Tutorial
==================

"""

######################################################################
# Uncomment this if you're using google colab to run this script
#

# !pip install pypose 


######################################################################
# ``LieTensor`` is the cornerstone of PyPose project. ``LieTensor`` is derived from
# ``torch.tensor``. it represents Lie Algebra or Lie Group. It support all 
# the ``torch.tensor`` features and also specific features for Lie Theory.
# 
# 
# We will see eventually in this tutorial that, with ``LieTensor``,
# one could easily implement operations often used in robotics applications.
#
# In PyPose, we would want to utilize the powerful network training API the comes with PyTorch.
# So, we will go a step further to see how we can use ``LieTensor`` in training a simple network.
#

import torch
import pypose as pp


######################################################################
# 1. Intialization
# ---------------------------------------
# The first thing we need to know is how to initialize a LieTensor.
# Use ``pypose.LieTensor`` or alias like ``pypose.so3``, specify the ``data`` and ``ltpye``.
# See list of ``ltype`` 
# `here <https://pypose.org/docs/main/generated/pypose.LieTensor/#pypose.LieTensor>`_.
#
# Note that the last dimension
# of ``data`` has to align with the ``LieTensor.ltype.dimension``
# because LieTensor has different length with respect to different ``ltype``.
# Here we have a ``(2,3)`` shaped tensor, because ``so3_type`` 
# requires a dimension of 3 for each element.
#
# It is recommanded to use alias to initialize LieTensor.


data = torch.randn(2, 3, requires_grad=True, device='cuda:0')
a = pp.LieTensor(data, ltype=pp.so3_type)
print('a:', a)
b = pp.so3(data)
print('b:', b)

######################################################################
# Like ``PyTorch``, you can initialize an identity ``LieTensor`` or a random ``LieTensor``.
# Use the function related to each ``ltype``. For example, here we used ``pypose.identity_SE3``
# and ``pypose.randn_se3``. The usage is similar with ``torch.randn``, except the shape we input
# is ``lshape``.
# The only difference between ``LieTensor.lshape`` and ``tensor.shape`` is the last dimension is hidden, since
# ``lshape`` takes the last dimension as a single ``ltype`` item.
# 


######################################################################
# You might notice the case difference here. 
# In PyPose, uppercase refers to Lie Group, and lowercase refers to Lie Algebra.
# It is recommanded to use Lie Group, unless Lie Algebra is absolutely necessary.
# 

######################################################################
# ``LieTensor.lview`` here is used to change the shape of a ``LieTensor``,
# similar to ``torch.view``.
# The difference is that ``LieTensor.lview`` does not modify the last dimension.
# It is intuitive since we need each element in ``x`` stays a ``SE3`` ltype.
# 

x = pp.identity_SE3(2,1)
y = pp.randn_se3(2,2)
print('x.shape:', x.shape, '\nx.gshape:', x.lshape)
print(x.lview(2))
print(y)


######################################################################
# 2. All arguments in PyTorch are supported
# ---------------------------------------------
# ``LieTensor`` is derived from ``torch.tensor``, so it inherit all the 
# attributes of a ``tensor``.
# You could specify ``device``, ``dtype``, and ``requires_grad`` during the initialization,
# just like PyTorch. 

a = pp.randn_SO3(3, device="cuda:0", dtype=torch.double, requires_grad=True)
b = pp.identity_like(a, device="cpu")
a, b

######################################################################
# And also, easy data type transform.

t = a.float()
a, t


######################################################################
# Slicing and Shaping
# ``LieTensor`` concatination is also the same as ``Pytorch``.

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
# 3. Exponential, Logarithm and Inversion Function
# ---------------------------------------------------
# ``LieTensor.Exp`` is the Exponential function defined in Lie Theory,
# which transform a input Lie Algebra to Lie Group. 
# ``LieTensor.Log`` is the Logarithm function, whcih transform Lie Group back to Lie Algebra.
# See the doc of
# `LieTensor.Exp <https://pypose.org/docs/main/generated/pypose.Exp/>`_ and 
# `LieTensor.Log <https://pypose.org/docs/main/generated/pypose.Log/>`_
# for the math.
#
# ``LieTensor.Inv`` gives us the inversion of a ``LieTensor``.
# Assume you have a ``LieTensor`` of ``pypose.so3_type``
# representing a rotation :math:`{\rm R}`, the `Inv` will give you :math:`{\rm R^{-1}}`.
# See `LieTensor.Inv <https://pypose.org/docs/main/generated/pypose.Inv/>`_.
# 


(x * y.Exp()).Inv().Log()


######################################################################
# 4. Adjoint Transforms
# ---------------------------------------
# We also have adjoint operations. Assume ``X`` is a Lie Group, 
# and ``a`` is a small left increment in Lie Algebra. 
# Adjoint operation will input ``a`` and output a right increment ``b`` that gives ther same transformation.
# See `pypose.Adj <https://pypose.org/docs/main/generated/pypose.Adj/>`_ for more details.
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
# As mentioned at the beginning, we would want to utilize the powerful
# network training API the comes with PyTorch.
# We might want to start by calculating gradients,
# which is a core step of any network training.
# First, we need to initialize the ``LieTensor`` of which we want to get gradients.
# Remember to set ``requires_grad=True``.

x = pp.randn_so3(3, sigma=0.1, requires_grad=True, device="cuda")
assert x.is_leaf

######################################################################
# And, just like in PyTorch, we will define a ``loss``, and call ``loss.backward``.
# That's it. Exactly the same with PyTorch.
# 


loss = (x.Exp().Log()**2).sin().sum() # Just test, No physical meaning
loss.backward()
y = x.detach()
loss, x.grad, x, y


######################################################################
# 6. Test a Module
# ---------------------------------------
# Now that we know all the basic operations, we might start ahead to build our first network.
# First of all, we define our ``TestNet`` as follows. Still, it doesn't have any physical meaning.
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


######################################################################
# Like PyTorch, we instantiate our network, optimizer, and scheduler.
# Scheduler here is to control the learning rate, see `lr_scheduler.MultiStepLR
# <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html#torch.optim.lr_scheduler.MultiStepLR>`_
# for more detail.
# 
# Then, inside the loop, we run our training. If you are not familiar with the training process,
# we would recommand you reading one of the PyTorch tutorial, like 
# `this <https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html>`_.
# 

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

######################################################################
# And then we are finished with our ``LieTensor`` tutorial.
# Hopefully you are more familiar with it by now.
# 
# Now you may be free to explore other tutorials. 
# See How PyPose can be utilized in real robotics applications.
# 
