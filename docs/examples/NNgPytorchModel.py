import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from itertools import chain
from tqdm import tqdm
import numpy as np


# loading data for training
xx = np.load('inputs_4sites_U2B4Nt32.npy')
yy = np.load('targets_4sites_U2B4Nt32.npy')

input_dim = xx.shape[1]
hidden_dim = [2*input_dim]
output_dim = input_dim


class NNg(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Args:
        input_dim: int for input dimension
        hidden_dim: list of int for multiple hidden layers
        output_dim: list of int for multiple output layers
        """

        # calling constructor of parent class
        super().__init__()

        # defining the inputs to the first hidden layer
        self.hid1 = nn.Linear(input_dim, hidden_dim[0])
        nn.init.normal_(self.hid1.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.hid1.bias)
        self.act1 = nn.ReLU()

        # defining the inputs to the output layer
        self.hid2 = nn.Linear(hidden_dim[0], output_dim)
        nn.init.xavier_uniform_(self.hid2.weight)

    def forward(self, X):

        # input and act for layer 1
        X = self.hid1(X)
        X = self.act1(X)

        # input and act for layer 2
        X = self.hid2(X)
        return X


epochs = 80
batchsize = 150
# to access a small section of the training data using the array indexing
inputs, targets = torch.from_numpy(xx).double(), torch.from_numpy(yy).double()
train_ds = TensorDataset(inputs, targets)
# split the data into batches
train_dl = DataLoader(train_ds, batchsize, shuffle=False)
test_dl = DataLoader(train_ds, batchsize, shuffle=False)
model_torch = NNg(input_dim, hidden_dim, output_dim).double()
optimizer = torch.optim.Adam(model_torch.parameters(), weight_decay=1e-5)
criterion = nn.L1Loss()

# iterate through all the epoch
for epoch in tqdm(range(epochs)):
    # go through all the batches generated by dataloader
    for i, (inputs, targets) in enumerate(train_dl):
        # clear the gradients
        optimizer.zero_grad()
        # compute the model output
        yhat = model_torch(inputs)
        # calculate loss
        loss = criterion(yhat, targets.type(torch.FloatTensor))
        # credit assignment
        loss.backward()
        # update model weights
        optimizer.step()
    # Print the progress
    if (epoch+1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch +
                                                   1, epochs, loss.item()))

# load a sample data
#example_input, example_target = next(iter(train_dl))
#run the tracing
#traced_script_module = torch.jit.trace(model_torch, example_input)
# save the converted model
#traced_script_module.save("NNg_model.pt")

#scripting the model
script_module = torch.jit.script(model_torch)
script_module.save("NNgModel_4sitesU2B4Nt32.pt")



