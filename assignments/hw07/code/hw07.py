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

print(y)

#check the data types
print("x:", x.shape, x.dtype, "y:", y.shape, y.dtype)
print("xtest:", xtest.shape, xtest.dtype, "ytest:", ytest.shape, ytest.dtype)

#Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#need to convert the data to tensors
x_tensor = torch.from_numpy(x)
xtest_tensor = torch.from_numpy(xtest)

#transposing 
y_tensor = torch.from_numpy(y).t().squeeze()
ytest_tensor = torch.from_numpy(ytest).t().squeeze()

#check if they've changed
print("After torch convers.  y:", y_tensor.shape, y_tensor.dtype, "ytest:", ytest_tensor.shape, ytest_tensor.dtype)

#initialising the loss function and the number of epochs to use
#as well as optimizer to use
epochs_to_use = 100
criterion = nn.CrossEntropyLoss()

#just picking one between 32 and 256
batch_size = 128

#need to convert data into Tensors
train_dataset = TensorDataset(x_tensor, y_tensor)
test_dataset = TensorDataset(xtest_tensor, ytest_tensor)

#DataLoaders
train_load = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_load = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

#part a.  1 hidden layer + ReLU 256 neurons
#defining model AND sending it to CUDA device
model = nn.Sequential(
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
).to(device)

#need model defined first.  hmmm  lr required, but documentation shows default is 1e-3
#https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html
optimizer = optim.SGD(model.parameters(), momentum=0.9, lr = 0.01)

#iterating over epochs.
#https://codesignal.com/learn/courses/building-a-neural-network-in-pytorch/lessons/training-a-neural-network-model-with-pytorch
#I think this dual loop works? https://discuss.pytorch.org/t/iterating-through-a-dataloader-object/25437
for epoch in range(epochs_to_use):

    #train model once per epoch
    model.train()

    for x_batch, y_batch in train_load:

        #send data to CUDA device
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        #set gradients to 0
        optimizer.zero_grad()

        #forward pass
        outputs = model(x_batch)

        #compute the loss
        loss = criterion(outputs, y_batch)

        #backward pass
        loss.backward()

        #optimize the model parameters
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")  

#predictions
model.eval()

with torch.no_grad():
    for x_batch, y_batch in test_load:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        #want predicted class labels
