"""
Get Started Tutorial
==================

"""

######################################################################
# Uncomment this if you're using google colab to run this script
#

# !pip install pypose 


######################################################################




######################################################################
# Sample Code of LieTensor
# ---------------------------------------
# The following code sample shows how to rotate random 
# points and compute the gradient of batched rotation.
#

import torch
import pypose as pp


######################################################################
# Create a random so(3) LieTensor
# 

r = pp.randn_so3(2, requires_grad=True)
print(r)


######################################################################
# Get the Lie Group of the Lie Algebra
# 

R = r.Exp() # Equivalent to: R = pp.Exp(r)
print(R)


######################################################################
# Create a random point and rotate it based on the Lie Group rotation tensor
# 

p = R @ torch.randn(3) # Rotate random point
print(p)


######################################################################
# Compute the gradient and print it
# 

p.sum().backward() # Compute gradient
r.grad # Print gradient


######################################################################
# Sample code of optimizer
# ---------------------------------------------
# We show how to estimate batched transform inverse by a
# 2nd-order optimizer. Two usage options for a scheduler
# are provided, each of which can work independently.
# 

from torch import nn
import torch, pypose as pp
from pypose.optim import LM
from pypose.optim.strategy import Constant
from pypose.optim.scheduler \
import StopOnPlateau

class InvNet(nn.Module):

    def __init__(self, *dim):
        super().__init__()
        init = pp.randn_SE3(*dim)
        self.pose = pp.Parameter(init)

    def forward(self, input):
        error = (self.pose @ input).Log()
        return error.tensor()
    
device = torch.device("cuda")
input = pp.randn_SE3(2, 2, device=device)
invnet = InvNet(2, 2).to(device)
strategy = Constant(damping=1e-4)
optimizer = LM(invnet, strategy=strategy)
scheduler = StopOnPlateau(optimizer,
                          steps=10,
                          patience=3,
                          decreasing=1e-3,
                          verbose=True)

# 1st option, full optimization
scheduler.optimize(input=input)

# 2nd option, step optimization
while scheduler.continual():
    loss = optimizer.step(input)
    scheduler.step(loss)



######################################################################
# And then we are finished with the two sample codes mentioned in our paper.
# 
# Now you may be free to explore other tutorials. 
# See How PyPose can be utilized in real robotics applications.
# 
