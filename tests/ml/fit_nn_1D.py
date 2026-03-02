#***************************************************
# Import Statements
#***************************************************
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from pcace.ml import MLP
from pcace.ml import afunc
import time

#***************************************************
# Define Device
#***************************************************

device = ("cpu")

#***************************************************
# Define Model
#***************************************************

model=MLP(
    n_in=1,n_out=1,
    n_hidden=[12,12],
    #activation=torch.nn.SiLU()
    #activation=afunc.IERF()
    #activation=afunc.SoftPlusZero()
    activation=afunc.SquarePlusZero()
)

#***************************************************
# Function Definitions
#***************************************************

#def func(x): return x * np.sin(x) # x*sin(x)
def func(x): return np.sin(np.pi*x)/x # sinc(x)

#***************************************************
# Define data
#***************************************************

# generate data
n_train=1000
n_test=100
batch_size = 32
x_train = np.linspace(start=-5, stop=5, num=n_train).reshape(-1, 1)
x_test = np.linspace(start=-5, stop=5, num=n_test).reshape(-1, 1)
y_train = func(x_train)
y_test = func(x_test)
train_data = TensorDataset(torch.Tensor(x_train),torch.Tensor(y_train))
test_data = TensorDataset(torch.Tensor(x_test),torch.Tensor(y_test))
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)

#***************************************************
# Batch
#***************************************************

def train_loop(dataloader, model, loss_fn, optimizer):
    # set training mode
    model.train()
    # loop over data in the batch
    for batch, (x, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(x)
        loss = loss_fn(pred, y)
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
def test_loop(dataloader, model, loss_fn):
    # set evaluation mode
    model.eval()
    # loop over data in the batch
    num_batches = len(dataloader)
    test_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    return test_loss

#***************************************************
# Training
#***************************************************

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print(model)

epochs = 1000
nprint = 10
start = time.time()
print("Epoch Loss")
for t in range(epochs):
    train_loop(train_loader, model, loss_fn, optimizer)
    test_loss=test_loop(test_loader, model, loss_fn)
    if(t%nprint==0): print(f"{t} {test_loss}")
print("Done!")
end = time.time()
print("time = ",end - start)

z_train=model(torch.Tensor(x_train)).detach().numpy().flatten()
z_test=model(torch.Tensor(x_test)).detach().numpy().flatten()
writer=open("predict_train.dat","w")
writer.write("#x y z\n")
for i in range(0,n_train):
    writer.write(str(x_train.flatten()[i])+" "+str(y_train.flatten()[i])+" "+str(z_train[i])+"\n")
writer.close()
writer=open("predict_test.dat","w")
writer.write("#x y z\n")
for i in range(0,n_test):
    writer.write(str(x_test.flatten()[i])+" "+str(y_test.flatten()[i])+" "+str(z_test[i])+"\n")
writer.close()
