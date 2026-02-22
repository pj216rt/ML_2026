import pandas as pd
import numpy as np
from scipy.io import loadmat
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt 

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

print(x_tensor[0].numel())

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

training_values = []
test_values = []

#iterating over epochs.
#https://codesignal.com/learn/courses/building-a-neural-network-in-pytorch/lessons/training-a-neural-network-model-with-pytorch
#I think this dual loop works? https://discuss.pytorch.org/t/iterating-through-a-dataloader-object/25437
for epoch in range(epochs_to_use):
    #Training
    #train model once per epoch
    model.train()
    train_correct = 0
    train_total = 0

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

        #add compute train accuracy
        outputs =  model(x_batch)
        soft_outputs = torch.nn.functional.softmax(outputs, dim=1)
        top_class = soft_outputs.topk(1, dim = 1)[1].squeeze()

        train_correct = train_correct + (top_class == y_batch).sum().item()
        train_total = train_total + y_batch.size(0)

    train_acc = train_correct / train_total

    #Testing now
    #predictions
    # #https://discuss.pytorch.org/t/obtain-probabilities-from-cross-entropy-loss/157259
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for x_batch, y_batch in test_load:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            #want predicted class labels
            outputs =  model(x_batch)
            soft_outputs = torch.nn.functional.softmax(outputs, dim=1)
            top_class = soft_outputs.topk(1, dim = 1)[1].squeeze()

            test_correct = test_correct + (top_class == y_batch).sum().item()
            test_total = test_total + y_batch.size(0)

    test_acc = test_correct / test_total

    #append values to empty lists
    training_values.append(train_acc)
    test_values.append(test_acc)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | Train acc: {train_acc:.4f} | Test acc: {test_acc:.4f}") 

#plotting this now
plt.figure()
plt.plot(range(epochs_to_use), training_values, label="Train Accuracy")
plt.plot(range(epochs_to_use), test_values, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train and Test Accuracy vs Epoch\nOne Hidden Layer, 256 neurons, ReLu Activation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



#part b.  two hidden layers, each with 64 neurons, ReLU activation
model = nn.Sequential(
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
).to(device)

optimizer = optim.SGD(model.parameters(), momentum=0.9, lr = 0.01)
training_values = []
test_values = []

#iterating over epochs.
#https://codesignal.com/learn/courses/building-a-neural-network-in-pytorch/lessons/training-a-neural-network-model-with-pytorch
#I think this dual loop works? https://discuss.pytorch.org/t/iterating-through-a-dataloader-object/25437
for epoch in range(epochs_to_use):
    #Training
    #train model once per epoch
    model.train()
    train_correct = 0
    train_total = 0

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

        #add compute train accuracy
        outputs =  model(x_batch)
        soft_outputs = torch.nn.functional.softmax(outputs, dim=1)
        top_class = soft_outputs.topk(1, dim = 1)[1].squeeze()

        train_correct = train_correct + (top_class == y_batch).sum().item()
        train_total = train_total + y_batch.size(0)

    train_acc = train_correct / train_total

    #Testing now
    #predictions
    # #https://discuss.pytorch.org/t/obtain-probabilities-from-cross-entropy-loss/157259
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for x_batch, y_batch in test_load:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            #want predicted class labels
            outputs =  model(x_batch)
            soft_outputs = torch.nn.functional.softmax(outputs, dim=1)
            top_class = soft_outputs.topk(1, dim = 1)[1].squeeze()

            test_correct = test_correct + (top_class == y_batch).sum().item()
            test_total = test_total + y_batch.size(0)

    test_acc = test_correct / test_total

    #append values to empty lists
    training_values.append(train_acc)
    test_values.append(test_acc)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | Train acc: {train_acc:.4f} | Test acc: {test_acc:.4f}") 

#plotting this now
plt.figure()
plt.plot(range(epochs_to_use), training_values, label="Train Accuracy")
plt.plot(range(epochs_to_use), test_values, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train and Test Accuracy vs Epoch\nTwo Hidden Layers, 64 neurons each, ReLu Activation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#part c.  one hidden layer, 32 filters of size 15.
#output length is Size of Data - Size of Kernel  + 1: 256-15+1
#Conv1d.  Issues with the dimensions.  
model = nn.Sequential(
    nn.Conv1d(in_channels=1, out_channels=32, kernel_size=15),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(32*242, 10)
).to(device)

optimizer = optim.SGD(model.parameters(), momentum=0.9, lr = 0.01)
training_values = []
test_values = []

#iterating over epochs.
#https://codesignal.com/learn/courses/building-a-neural-network-in-pytorch/lessons/training-a-neural-network-model-with-pytorch
#I think this dual loop works? https://discuss.pytorch.org/t/iterating-through-a-dataloader-object/25437
for epoch in range(epochs_to_use):
    #Training
    #train model once per epoch
    model.train()
    train_correct = 0
    train_total = 0

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

        #add compute train accuracy
        outputs =  model(x_batch)
        soft_outputs = torch.nn.functional.softmax(outputs, dim=1)
        top_class = soft_outputs.topk(1, dim = 1)[1].squeeze()

        train_correct = train_correct + (top_class == y_batch).sum().item()
        train_total = train_total + y_batch.size(0)

    train_acc = train_correct / train_total

    #Testing now
    #predictions
    # #https://discuss.pytorch.org/t/obtain-probabilities-from-cross-entropy-loss/157259
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for x_batch, y_batch in test_load:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            #want predicted class labels
            outputs =  model(x_batch)
            soft_outputs = torch.nn.functional.softmax(outputs, dim=1)
            top_class = soft_outputs.topk(1, dim = 1)[1].squeeze()

            test_correct = test_correct + (top_class == y_batch).sum().item()
            test_total = test_total + y_batch.size(0)

    test_acc = test_correct / test_total

    #append values to empty lists
    training_values.append(train_acc)
    test_values.append(test_acc)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | Train acc: {train_acc:.4f} | Test acc: {test_acc:.4f}") 

#plotting this now
plt.figure()
plt.plot(range(epochs_to_use), training_values, label="Train Accuracy")
plt.plot(range(epochs_to_use), test_values, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train and Test Accuracy vs Epoch\n CNN One Hidden Layer, 32 filters, Size 15, ReLu Activation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()