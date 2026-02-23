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

#need to cut down the number of lines of code.  800 is insane
#a single .mat file
mat = loadmat("assignments/hw07/data/cnnslm3000.mat")

#one liners to get x and x test.  stores them as tensors from the getgo
x, xtest = (torch.from_numpy(mat[k]) for k in ("x", "xtest"))
y, ytest = (torch.from_numpy(mat[k]).t().squeeze() for k in ("y", "ytest"))

#check the data types.  mostly for debugging. 
#print("x:", x.shape, x.dtype, "y:", y.shape, y.dtype)
#print("xtest:", xtest.shape, xtest.dtype, "ytest:", ytest.shape, ytest.dtype)

#Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#initialising the loss function and the number of epochs to use
#as well as optimizer to use
epochs_to_use = 100
criterion = nn.CrossEntropyLoss()

#just picking one between 32 and 256
batch_size = 128

#need to get Dataloaders of the Datasets
train_load = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
test_load  = DataLoader(TensorDataset(xtest, ytest), batch_size=batch_size, shuffle=False)

#function for reuse
#training the data.
def run_NN_train(model, loader, optimizer, criterion, device):
    model.train()
    correct, total = 0, 0

    #batchs
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        #set gradients to 0, forward pass, compute the losee
        #backward pass, then update model parameters
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        #make predictions
        soft_outputs = torch.nn.functional.softmax(outputs, dim=1)
        top_class = soft_outputs.topk(1, dim = 1)[1].squeeze()

        correct = correct + (top_class == y_batch).sum().item()
        total = total + y_batch.size(0)

    return (correct/total)

#function for testing the data.  Don't need optimizer anymore?
def run_NN_test(model, loader, device):
    model.eval()
    correct, total = 0,0

    #batches
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            #want predicted class labels
            outputs =  model(x_batch)
            soft_outputs = torch.nn.functional.softmax(outputs, dim=1)
            top_class = soft_outputs.topk(1, dim = 1)[1].squeeze()

            correct = correct + (top_class == y_batch).sum().item()
            total = total + y_batch.size(0)
            
        return (correct/total)

#function to train and plot
def run_all(model, training_data, testing_data, device, epochs=100, lr=0.01,
            momentum=0.9, print_option = 10, image_saved_path=None, plot_title=""):
    #send model to CUDA device, setting optimizer, and criterion
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    train_accs, test_accs = [], []
    for epoch in range(epochs):
        #for each epoch, train and test
        train_acc = run_NN_train(model, train_load, optimizer, criterion=criterion, device=device)
        test_acc  = run_NN_test(model, test_load, device)

        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if (epoch + 1) % print_option == 0:
            print(f"Epoch {epoch+1:3d} | Train acc: {train_acc:.4f} | Test acc: {test_acc:.4f}")

    #plotting stuff
    plt.figure()
    plt.plot(range(epochs), train_accs, label="Train Accuracy")
    plt.plot(range(epochs), test_accs, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(plot_title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(image_saved_path, dpi=400, bbox_inches="tight")
    plt.close()

    return train_accs, test_accs

#information for a loop.  List of Dicts
parts_a_b = [
    dict(
        name="part_a",
        model=nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        ),
        title="Train and Test Accuracy vs Epoch\nOne Hidden Layer, 256 neurons, ReLU Activation",
        save_path="assignments/hw07/figures/part_a.png"
    ),
    dict(
        name="part_b",
        model=nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        ),
        title="Train and Test Accuracy vs Epoch\nTwo Hidden Layers, 64 neurons each, ReLU Activation",
        save_path="assignments/hw07/figures/part_b.png"
    )
]


#parts a and b
results = []
for item in parts_a_b:
    print(item["name"])
    train_accuracy, test_accuracy = run_all(
        model=item["model"],
        training_data=train_load, testing_data=test_load,
        device=device, plot_title=item["title"], image_saved_path=item["save_path"], print_option=25
    )

    #add to the results object the final train and test accuracies
    results.append({
        "model": item["name"],
        "final_train_acc": train_accuracy[-1],
        "final_test_acc": test_accuracy[-1]
    })



def run_CNN_train(model, loader, optimizer, criterion, device):
    model.train()
    correct, total = 0, 0

    #batchs
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        x_batch = x_batch.unsqueeze(1)
        y_batch = y_batch.to(device)

        #set gradients to 0, forward pass, compute the losee
        #backward pass, then update model parameters
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        #make predictions
        soft_outputs = torch.nn.functional.softmax(outputs, dim=1)
        top_class = soft_outputs.topk(1, dim = 1)[1].squeeze()

        correct = correct + (top_class == y_batch).sum().item()
        total = total + y_batch.size(0)

    return (correct/total)

#function for testing the data.  Don't need optimizer anymore?
def run_CNN_test(model, loader, device):
    model.eval()
    correct, total = 0,0

    #batches.  Change here as opposed to the NN
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            x_batch = x_batch.unsqueeze(1)
            y_batch = y_batch.to(device)

            #want predicted class labels
            outputs =  model(x_batch)
            soft_outputs = torch.nn.functional.softmax(outputs, dim=1)
            top_class = soft_outputs.topk(1, dim = 1)[1].squeeze()

            correct = correct + (top_class == y_batch).sum().item()
            total = total + y_batch.size(0)

        return (correct/total)

#function to train and plot
def run_all_CNN(model, training_data, testing_data, device, epochs=100, lr=0.01,
            momentum=0.9, print_option = 10, image_saved_path=None, plot_title=""):
    #send model to CUDA device, setting optimizer, and criterion
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    train_accs, test_accs = [], []
    for epoch in range(epochs):
        #for each epoch, train and test
        train_acc = run_CNN_train(model, train_load, optimizer, criterion=criterion, device=device)
        test_acc  = run_CNN_test(model, test_load, device)

        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if (epoch + 1) % print_option == 0:
            print(f"Epoch {epoch+1:3d} | Train acc: {train_acc:.4f} | Test acc: {test_acc:.4f}")

    #plotting stuff
    plt.figure()
    plt.plot(range(epochs_to_use), train_accs, label="Train Accuracy")
    plt.plot(range(epochs_to_use), test_accs, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(plot_title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(image_saved_path, dpi=400, bbox_inches="tight")
    plt.close()

    return train_accs, test_accs

parts_c_g = [
    dict(
        name="part_c",
        model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=15),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*242, 10)
        ),
        title="Train and Test Accuracy vs Epoch\n CNN One Hidden Layer, 32 filters, Size 15, ReLu Activation",
        save_path="assignments/hw07/figures/part_c.png"
    ),
    dict(
        name="part_d",
        model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=15),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(10)
        ),
        title="Train and Test Accuracy vs Epoch\n CNN One Hidden Layer, 32 filters, Size 15, ReLu Activation, Max Pool=2, Stride 2",
        save_path="assignments/hw07/figures/part_d.png"
    ),
    dict(
        name="part_e",
        model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=15),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=242, stride=242),
            nn.Flatten(),
            nn.Linear(32, 10)
        ),
        title="Train and Test Accuracy vs Epoch\n CNN One Hidden Layer, 32 filters, Size 15, ReLu Activation, Max Pooling",
        save_path="assignments/hw07/figures/part_e.png"
    ),
    dict(
        name="part_f",
        model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=15),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=15),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.LazyLinear(10)
        ),
        title="Train and Test Accuracy vs Epoch\n CNN Two Hidden Layers, 16 filters, Size 15, ReLu Activation\nMax Pooling, Size 2, Stride 2",
        save_path="assignments/hw07/figures/part_f.png"
    ),
    dict(
        name="part_g",
        model = nn.Sequential(
            #layer 1
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=15),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            #layer 2
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=15),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            #layer 3
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=15),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.LazyLinear(10)
        ),
        title="Train and Test Accuracy vs Epoch\n CNN Three Hidden Layers, 16 filters, Size 15, ReLu Activation\nMax Pooling, Size 2, Stride 2",
        save_path="assignments/hw07/figures/part_g.png"
    )
]

#parts c through g
results_cnn = []
for item in parts_c_g:
    print(item["name"])
    train_accuracy, test_accuracy = run_all_CNN(
        model=item["model"],
        training_data=train_load, testing_data=test_load,
        device=device, plot_title=item["title"], image_saved_path=item["save_path"], print_option=25
    )

    #add to the results object the final train and test accuracies
    results_cnn.append({
        "model": item["name"],
        "final_train_acc": train_accuracy[-1],
        "final_test_acc": test_accuracy[-1]
    })

#part h.  One big table
total_results = pd.DataFrame(results + results_cnn)
print(total_results)