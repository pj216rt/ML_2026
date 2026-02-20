import pandas as pd
import numpy as np
from scipy.io import loadmat

#loading in torch 
#https://pytorch.org/get-started/locally/
#https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
import torch
from torch import nn

#a single .mat file
mat = loadmat("assignments/hw07/data/cnnslm3000.mat")

#saving objects
x = mat["x"]
y = mat["y"]
xtest = mat["xtest"]
ytest = mat["ytest"]

#part a
print(torch.cuda.is_available())

#which device are we using
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")