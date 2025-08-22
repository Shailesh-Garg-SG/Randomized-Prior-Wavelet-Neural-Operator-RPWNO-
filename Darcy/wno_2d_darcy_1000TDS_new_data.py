import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.cuda.empty_cache()
import matplotlib.pyplot as plt

from timeit import default_timer
from utilities3 import UnitGaussianNormalizer, count_params, LpLoss, MatReader
from pytorch_wavelets import DWT, IDWT # (or import DWT, IDWT)

torch.manual_seed(0)
np.random.seed(0)

# %%

""" Def: 2d Wavelet layer """

class WaveConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(WaveConv2d, self).__init__()

        """
        2D Wavelet layer. It does DWT, linear transform, and Inverse dWT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 # Number of Wavelet level to multiply, at most floor(N_s/2**level) + (2*N_w-2), for db(N_w) Wavelet
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))

    # Convolution
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute single tree Discrete Wavelet coefficients using some wavelet
        dwt = DWT(J=4, mode='symmetric', wave='db4').cuda()
        x_ft, x_coeff = dwt(x)

        # Multiply relevant Wavelet modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x_ft.shape[-2], x_ft.shape[-1], device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        # Multiply the finer wavelet coefficients
        x_coeff[-1][:,:,0,:,:] = self.compl_mul2d(x_coeff[-1][:,:,0,:,:].clone(), self.weights2)
        
        # Return to physical space        
        idwt = IDWT(mode='symmetric', wave='db4').cuda()
        x = idwt((out_ft, x_coeff))
        return x

""" The forward operation """

class WNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(WNO2d, self).__init__()

        """
        The WNO network. It contains 4 layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. 4 layers of the integral operators v(+1) = g(K(.) + W)(v).
            W is defined by self.w_; K is defined by self.conv_.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 1 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = WaveConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = WaveConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = WaveConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = WaveConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 192)
        self.fc2 = nn.Linear(192, 1)
        
        self.fc0_NT = nn.Linear(3, self.width).requires_grad_(False) # input channel is 3: (a(x, y), x, y)

        self.conv0_NT = WaveConv2d(self.width, self.width, self.modes1, self.modes2).requires_grad_(False)
        self.conv1_NT = WaveConv2d(self.width, self.width, self.modes1, self.modes2).requires_grad_(False)
        self.conv2_NT = WaveConv2d(self.width, self.width, self.modes1, self.modes2).requires_grad_(False)
        self.conv3_NT = WaveConv2d(self.width, self.width, self.modes1, self.modes2).requires_grad_(False)
        self.w0_NT = nn.Conv2d(self.width, self.width, 1).requires_grad_(False)
        self.w1_NT = nn.Conv2d(self.width, self.width, 1).requires_grad_(False)
        self.w2_NT = nn.Conv2d(self.width, self.width, 1).requires_grad_(False)
        self.w3_NT = nn.Conv2d(self.width, self.width, 1).requires_grad_(False)

        self.fc1_NT = nn.Linear(self.width, 192).requires_grad_(False)
        self.fc2_NT = nn.Linear(192, 1).requires_grad_(False)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        # print(x.shape, grid.shape)
        x = torch.cat((x, grid), dim=-1)
        # print(x.shape, grid.shape)
        
         
         
         
        
        x_NT = self.fc0_NT(x)
        x_NT = x_NT.permute(0, 3, 1, 2)
        x_NT = F.pad(x_NT, [0,self.padding,0,self.padding]) # do padding, if required
        
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
        
        x_NT = x_NT[..., :-self.padding, :-self.padding] # remove padding, when required
        x_NT = x_NT.permute(0, 2, 3, 1)
        x_NT = self.fc1_NT(x_NT)
        x_NT = F.gelu(x_NT)
        x_NT = self.fc2_NT(x_NT)
        
        
        
        
        
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding]) # padding, if required
        
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
        
        x = x[..., :-self.padding, :-self.padding] # removing padding, when applicable
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        x = x+1*x_NT
        
        return x
    
    def get_grid(self, shape, device):
        # The grid of the solution
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

# %%

""" Model configurations """

# /DATA/SG/WNO/data/
TRAIN_PATH = 'E:\WNO\data\darcy_2d_data_1_1000.mat'
TEST_PATH = 'E:\WNO\data\darcy_2d_data_1001_2000.mat'
ntrain = 1000
ntest = 40

batch_size = 20
learning_rate = 0.001

epochs = 500
step_size = 50
gamma = 0.75

modes = 11
width = 64

r = 5
h = int(((421 - 1)/r) + 1)
s = h

# %%

""" Read data """

reader = MatReader(TRAIN_PATH)
x_train = reader.read_field('in')[:ntrain,::r,::r][:,:s,:s]
y_train = reader.read_field('out')[:ntrain,::r,::r][:,:s,:s]

reader.load_file(TEST_PATH)
x_test = reader.read_field('in')[:ntest,::r,::r][:,:s,:s]
y_test = reader.read_field('out')[:ntest,::r,::r][:,:s,:s]

x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

x_train = x_train.reshape(ntrain,s,s,1)
x_test = x_test.reshape(ntest,s,s,1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

# %%

""" Sample model definition """

model = WNO2d(modes, modes, width).cuda()
print(count_params(model))

from torchinfo import summary
print(summary(model, input_size=(batch_size, 85, 85, 1)))

# %%

""" Training and testing """

import pickle

y_normalizer.cuda()

for model_num in range (0,10):
    model = WNO2d(modes, modes, width).cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    myloss = LpLoss(size_average=False)
    y_normalizer.cuda()
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
    
            optimizer.zero_grad()
            out = model(x).reshape(batch_size, s, s)
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)
            
            loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
            loss.backward()
            optimizer.step()
            train_l2 += loss.item()
        
        scheduler.step()
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
    
                out = model(x).reshape(batch_size, s, s)
                out = y_normalizer.decode(out)
    
                test_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()
    
        train_l2/= ntrain
        test_l2 /= ntest
        t2 = default_timer()
        print(model_num, ep, t2-t1, train_l2, test_l2)
        # print(model_num, ep, t2-t1, train_l2)
                
    filename = './RPmodel/new_data_finalized_model_FINAL2D_1000TDS_'+str(model_num)+'.sav'
    pickle.dump(model, open(filename, 'wb'))
        
# %%

""" Prediction """

s = h

ntest = 8000
x_test = torch.zeros([ntest,s,s])
y_test = torch.zeros([ntest,s,s])

for i in range(2,10):
    print(i)
    TEST_PATH = 'E:\WNO\data\darcy_2d_data_'+str(i*1000+1)+'_'+str((i+1)*1000)+'.mat'

    reader.load_file(TEST_PATH)
    x_test[(i-2)*1000:(i+1-2)*1000] = reader.read_field('in')[:,::r,::r][:,:s,:s]
    y_test[(i-2)*1000:(i+1-2)*1000] = reader.read_field('out')[:,::r,::r][:,:s,:s]

x_test = x_normalizer.encode(x_test)
y_train = y_normalizer.encode(y_train)

x_test = x_test.reshape(ntest,s,s,1)

bs = 50

test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test),
                                          batch_size=bs, shuffle=False)

""" Prediction """

y_normalizer.cuda()
pred = torch.zeros([10, y_test.shape[0], y_test.shape[1], y_test.shape[2]])
# myloss = LpLoss(size_average=False)

for model_num in range (0,10):

    print(model_num)
    filename = './RPmodel/finalized_model_FINAL2D_1000TDS_'+str(model_num)+'.sav' ## TRIAL 103
    loaded_model = pickle.load(open(filename, 'rb'))

    index = 0
    test_e = torch.zeros(y_test.shape[0])
    
    with torch.no_grad():
        for x, y in test_loader:
            t1 = default_timer()
            test_l2 = 0
            x, y = x.cuda(), y.cuda()
    
            out = loaded_model(x).reshape(bs,s, s)
            out = y_normalizer.decode(out)
            pred[model_num,index*bs:(index+1)*bs,:,:] = out
    
            test_l2 = myloss(out.reshape(1, s, s), y.reshape(1, s, s)).item()
            test_e[index] = test_l2
            
            print(index, test_l2)
            
            t2 = default_timer()
            print(model_num, index, t2-t1)
            
            index = index + 1
            
        print('Mean Testing Error:', 100*torch.mean(test_e).numpy(), '%')
        

# %%

m = np.mean(pred.numpy(),0)
std = np.std(pred.numpy(),0)

plt.figure(constrained_layout=False, figsize = (7, 14))
plt.subplots_adjust(hspace=0.75)
index = 0

for value in range(y_test.shape[0]):
    if value % 6000 == 1:
        plt.subplot(5,2, index+1)
        plt.imshow(y_test[value,:,:], origin='lower', cmap='seismic',
                   vmin = np.min(y_test[value,:,:].numpy()), vmax = np.max(y_test[value,:,:].numpy()))
        plt.colorbar()
        plt.title('Actual')
                
        plt.subplot(5,2, index+1+2)
        plt.imshow(m[value,:,:], origin='lower', cmap='seismic',
                   vmin = np.min(y_test[value,:,:].numpy()), vmax = np.max(y_test[value,:,:].numpy()))
        plt.colorbar()
        plt.title('Mean')
                
        plt.subplot(5,2, index+1+4)
        plt.imshow(std[value,:,:], origin='lower', cmap='seismic')
        plt.colorbar()
        plt.title('Std')
                
        plt.subplot(5,2, index+1+6)
        plt.imshow(m[value,:,:]-1.96*std[value,:,:], origin='lower', cmap='seismic',
                   vmin = np.min(y_test[value,:,:].numpy()), vmax = np.max(y_test[value,:,:].numpy()))
        plt.colorbar()
        plt.title('Lower Limit')
        
        plt.subplot(5,2, index+1+8)
        plt.imshow(m[value,:,:]-1.96*std[value,:,:], origin='lower', cmap='seismic',
                   vmin = np.min(y_test[value,:,:].numpy()), vmax = np.max(y_test[value,:,:].numpy()))
        plt.colorbar()
        plt.title('Lower Limit')
                
        # plt.margins(0)
        index = index + 1
        
print((100*myloss(y_test, torch.tensor(m)).item())/40)
print(100*torch.mean((y_test-torch.tensor(m))**2).item()/torch.mean((y_test**2)).item())

# %%

for marker in ([1-1,5-1,15-1,30-1,50-1,75-1,85-1]):
    plt.figure(constrained_layout=False, figsize = (10, 3))
    plt.subplots_adjust(wspace=0.3)
    index = 0
    for value in range(y_test.shape[0]):
        if value % 6000 == 1:
            plt.subplot(1,2, index+1)
            plt.plot(y_test[value,marker,:],'r')
            plt.plot(m[value,marker,:],'b')
            plt.plot(m[value,marker,:]+1.96*std[value,marker,:],'m')
            plt.plot(m[value,marker,:]-1.96*std[value,marker,:],'m')

            # plt.margins(0)
            index = index + 1

# %%

# torch.cuda.empty_cache()

import scipy.io as sio
sio.savemat('data_1000_new_data_code.mat',{'pred':pred.numpy(), 'y_test':y_test.numpy()})
