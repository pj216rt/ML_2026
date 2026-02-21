import pandas as pd
import numpy as np
from scipy.io import loadmat
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim 

#https://pytorch.org/get-started/locally/
#https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
#https://www.codecademy.com/article/building-a-neural-network-using-pytorch


#a single .mat file
mat = loadmat("assignments/hw07/data/cnnslm3000.mat")

#saving objects
x = mat["x"]
y = mat["y"]
xtest = mat["xtest"]
ytest = mat["ytest"]
print("x:", x.shape, x.dtype, "y:", y.shape, y.dtype)

#initialising the loss function and the number of epochs to use
#as well as optimizer to use
epochs_to_use = 100
criterion = nn.CrossEntropyLoss()
batch_size = 128

#need to convert data into Tensors
train_ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
test_ds  = TensorDataset(torch.from_numpy(xtest), torch.from_numpy(ytest))

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

#part a
#defining the class
class NN_1_hidden_256_ReLU(nn.Module):
    def __init__(self): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(),
            nn.ReLU(),
            nn.Linear(256)
        ) 

    def forward(self, x): 
        return self.net(x)

model = NN_1_hidden_256_ReLU()
print(model)

optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=0.01)

for epoch in range(epochs_to_use):
    optimizer.zero_grad()
    outputs = model(x)

    #compute loss
    loss = criterion(outputs, y)

    #backaward pass for gradients
    loss.backward()

    #update weights
    optimizer.step()
    print(f'Epoch [{epoch + 1}/5], Loss: {loss.item():.4f}')

#evaluating the model
prediction = model(xtest) 
print(f'Prediction: {prediction.item():.4f}') 