from IPython import get_ipython
get_ipython().magic('reset -sf')

# import pywt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.cuda.empty_cache()
import matplotlib.pyplot as plt

from timeit import default_timer
from utilities3 import *
from pytorch_wavelets import DWT1D, IDWT1D

torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%

""" Def: 1d Wavelet layer """

class WaveConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, level1):
        super(WaveConv1d, self).__init__()

        """
        1D Wavelet layer. It does Wavelet Transform, linear transform, and
        Inverse Wavelet Transform.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level1 = level1  # Number of Wavelet level to multiply, at most floor(N_s/2**level) + (2*N_w-2), for db(N_w) Wavelet

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.level1+6))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.level1+6))

    # Convolution
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute single tree Discrete Wavelet coefficients using some wavelet     
        dwt = DWT1D(wave='db6', J=self.level1, mode='symmetric').to(device)
        x_ft, x_coeff = dwt(x)
        
        # Multiply the final low pass and high pass coefficients
        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.shape[-1],  device=x.device)
        out_ft = self.compl_mul1d(x_ft, self.weights1)
        x_coeff[-1] = self.compl_mul1d(x_coeff[-1], self.weights2)
        
        # Reconstruct the signal
        idwt = IDWT1D(wave='db6', mode='symmetric').to(device)
        x = idwt((out_ft, x_coeff))        
        return x

""" The forward operation """

class WNO1d(nn.Module):
    def __init__(self, level, width):
        super(WNO1d, self).__init__()

        """
        The WNO network. It contains 4 layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. 4 layers of the integral operators v(+1) = g(K(.) + W)(v).
            W is defined by self.w_; K is defined by self.conv_.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.level1 = level
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)

        self.conv0 = WaveConv1d(self.width, self.width, self.level1)
        self.conv1 = WaveConv1d(self.width, self.width, self.level1)
        self.conv2 = WaveConv1d(self.width, self.width, self.level1)
        self.conv3 = WaveConv1d(self.width, self.width, self.level1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        self.fc0_NT = nn.Linear(2, self.width).requires_grad_(False)

        self.conv0_NT = WaveConv1d(self.width, self.width, self.level1).requires_grad_(False)
        self.conv1_NT = WaveConv1d(self.width, self.width, self.level1).requires_grad_(False)
        self.conv2_NT = WaveConv1d(self.width, self.width, self.level1).requires_grad_(False)
        self.conv3_NT = WaveConv1d(self.width, self.width, self.level1).requires_grad_(False)
        self.w0_NT = nn.Conv1d(self.width, self.width, 1).requires_grad_(False)
        self.w1_NT = nn.Conv1d(self.width, self.width, 1).requires_grad_(False)
        self.w2_NT = nn.Conv1d(self.width, self.width, 1).requires_grad_(False)
        self.w3_NT = nn.Conv1d(self.width, self.width, 1).requires_grad_(False)

        self.fc1_NT = nn.Linear(self.width, 128).requires_grad_(False).requires_grad_(False)
        self.fc2_NT = nn.Linear(128, 1).requires_grad_(False).requires_grad_(False)

    def forward(self, x):
        
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        
        
        
        
        
        x_NT = self.fc0_NT(x)
        x_NT = x_NT.permute(0, 2, 1)
        # x_NT = F.pad(x_NT, [0,self.padding]) # do padding, if required

        x1_NT = self.conv0_NT(x_NT)
        x2_NT = self.w0_NT(x_NT)
        x_NT = x1_NT + x2_NT
        x_NT = F.gelu(x_NT)

        x1_NT = self.conv1_NT(x_NT)
        x2_NT = self.w1_NT(x_NT)
        x_NT = x1_NT + x2_NT
        x_NT = F.gelu(x_NT)

        x1_NT = self.conv2_NT(x_NT)
        x2_NT = self.w2_NT(x_NT)
        x_NT = x1_NT + x2_NT
        x_NT = F.gelu(x_NT)

        x1_NT = self.conv3_NT(x_NT)
        x2_NT = self.w3_NT(x_NT)
        x_NT = x1_NT + x2_NT

        # x_NT = x_NT[..., :-self.padding] # remove padding, when required
        x_NT = x_NT.permute(0, 2, 1)
        x_NT = self.fc1_NT(x_NT)
        x_NT = F.gelu(x_NT)
        x_NT = self.fc2_NT(x_NT)
        
        
        
        
        
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) # do padding, if required

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding] # remove padding, when required
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        
        
        
        
        x = x+1*x_NT ## TRIAL 103
        
        return x

    def get_grid(self, shape, device):
        # The grid of the solution
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)


# %%

""" Model configurations """

ntrain = 1000
ntest = 8000

sub = 2**3 # subsampling rate
h = 2**13 // sub # total grid size divided by the subsampling rate
s = h

batch_size = 10
learning_rate = 0.001

epochs = 500
step_size = 100
gamma = 0.5

level = 8 # The automation of the mode size will made and shared upon acceptance and final submission
width = 64

# %%

""" Read data """

# Data is of the shape (number of samples, grid size)
import scipy.io as sio

dataloader = sio.loadmat('E:/WNO/data/burger_v_0p01.mat')

x_data = torch.tensor(dataloader['input'][:,::sub], dtype=torch.float)
y_data = torch.tensor(dataloader['output'][:,::sub], dtype=torch.float)

x_train = x_data[:ntrain,:].to(device)
y_train = y_data[:ntrain,:]
x_test = x_data[-ntest:,:]
y_test = y_data[-ntest:,:]

x_train = x_train.reshape(ntrain,s,1)
x_test = x_test.reshape(ntest,s,1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

# %%

# """ The model definition """

model = WNO1d(level, width).to(device)
print(count_params(model))

# from torchinfo import summary
# print(summary(model, input_size=(batch_size, 1024, 1)))

# from torchviz import make_dot
# make_dot(model(x_train), params=dict(model.named_parameters())).render("torchviz", format="png")

# %%

# """ Training and testing """

import pickle

for model_num in range (0,10):
    
    model = WNO1d(level, width).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    myloss = LpLoss(size_average=False)
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
    
            optimizer.zero_grad()
            out = model(x)
            
            mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
            l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            l2.backward() # l2 relative loss
    
            optimizer.step()
            train_mse += mse.item()
            train_l2 += l2.item()
    
        scheduler.step()
        # model.eval()
        # test_l2 = 0.0
        # with torch.no_grad():
        #     for x, y in test_loader:
        #         x, y = x.to(device), y.to(device)
    
        #         out = model(x)
        #         test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()
    
        train_mse /= len(train_loader)
        train_l2 /= ntrain
        # test_l2 /= ntest
    
        t2 = default_timer()
        
        # if ep%50 == 0:
        #     print(model_num, ep, t2-t1, train_mse, train_l2, test_l2)
        if ep%50 == 0:
            print(model_num, ep, t2-t1, train_mse, train_l2)
        
    filename = './RPmodel/new_data_finalized_model_FINAL1D_'+str(model_num)+'.sav' ## TRIAL 103
    pickle.dump(model, open(filename, 'wb'))
     
# %%

""" Prediction """

import pickle

pred = torch.zeros([10, y_test.shape[0], y_test.shape[1]])
myloss = LpLoss(size_average=False)

for model_num in range (0,10):

    filename = './RPmodel/new_data_finalized_model_FINAL1D_'+str(model_num)+'.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    
    
    index = 0
    test_e = torch.zeros(y_test.shape[0])
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
    
    with torch.no_grad():
        for x, y in test_loader:
            test_l2 = 0
            x, y = x.to(device), y.to(device)
    
            out = loaded_model(x).view(-1)
            pred[model_num, index,:] = out
    
            test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
            test_e[index] = test_l2
            
            # print(index, test_l2)
            index = index + 1
    
    print(model_num, 'Mean Error:', 100*torch.mean(test_e))

# %%

""" Plotting """

m = np.mean(pred.numpy(),0)
s = np.std(pred.numpy(),0)

figure7 = plt.figure(figsize = (10, 8))
for i in range(50,y_test.shape[0]):
    if i % 6000 == 1:
        plt.plot(y_test[i, :].numpy(), 'r', label='Actual')
        plt.plot(m[i,:], 'k', label='Prediction')
        
        predsamp_ul = m[i, :]+1.96*s[i, :]
        predsamp_ll = m[i, :]-1.96*s[i, :]

        plt.fill_between(np.arange(0,1024,1), predsamp_ul, predsamp_ll, color = 'c', alpha = 0.5)
        
plt.legend()
plt.margins(0)


import scipy.io as sio
# sio.savemat('Burgers1000TDS.mat',{'m':m, 's':s, 'y':y_test.numpy()})

sio.savemat('new_data_Burgers1000TDS.mat',{'m':m, 's':s, 'y':y_test.numpy(), 'p':pred.numpy()})







