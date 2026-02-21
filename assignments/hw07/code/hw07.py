import pandas as pd
import numpy as np
from scipy.io import loadmat

#loading in torch 
#https://pytorch.org/get-started/locally/
#https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
import torch
from torch import nn

#checking torch stuff
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))

#a single .mat file
mat = loadmat("assignments/hw07/data/cnnslm3000.mat")

#saving objects
x = mat["x"]
y = mat["y"]
xtest = mat["xtest"]
ytest = mat["ytest"]

#part a
print("x:", x.shape, x.dtype, "y:", y.shape, y.dtype)