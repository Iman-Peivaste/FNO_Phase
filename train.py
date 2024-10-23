#This is for training the FNO for multi grain
#%%
import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parameter as Parameter
import matplotlib.pyplot as plt
import operator
import matplotlib.pyplot as plt
from functools import reduce
from functools import partial 
from timeit import default_timer
from utilities3 import *
from Adam import Adam
from tqdm import tqdm
import gc
import os
import pickle
gc.collect()
torch.cuda.empty_cache()
torch.manual_seed(0)
np.random.seed(0)
#%%
shift_number = 10
Test_num = 200
Test_raio = 0.1
T = 10
dim = 64
stride = 9
#%%

Dataset2 = np.load('Dataset_b_64.npy')

print(Dataset2.shape)

#%%
print(Dataset2.shape)

# Create input-output pairs with a time shift
inp = Dataset2[:, 0 : Dataset2.shape[1] - shift_number, :, :]
out = Dataset2[:, shift_number : Dataset2.shape[1], :, :]

# Calculate the number of sequences per sample
num_samples = inp.shape[0]
num_time_steps = inp.shape[1]
num_sequences_per_sample = (num_time_steps - T) // stride + 1



# Initialize lists to collect sequences
xx_list = []
yy_list = []

# Extract overlapping sequences
for sample_idx in range(num_samples):
    for seq_idx in range(num_sequences_per_sample):
        start = seq_idx * stride
        end = start + T
        xx_list.append(inp[sample_idx, start:end, :, :])
        yy_list.append(out[sample_idx, start:end, :, :])

# Convert lists to arrays
xx = np.stack(xx_list, axis=0)
yy = np.stack(yy_list, axis=0)

print(f'Input shape: {xx.shape}')  # Should be (total_sequences, T, dim, dim)
print(f'Output shape: {yy.shape}')

#%%
# x=[]
# y=[]
# for i in range (xx.shape[0]):
#     t = xx[i,:,:,:]
#     z = yy[i,:,:,:]
#     comparison = t==z
#     equal_arrays = comparison.all()
#     if equal_arrays:
#         continue
#     else:
        
#         x.append(t)
#         y.append(z)

# x = np.array(x)
# y = np.array(y)
# Set a tolerance level for floating-point comparison
tolerance = 1e-6

# Calculate the absolute difference between xx and yy
difference = np.abs(xx - yy)

# Create a boolean mask where sequences are considered equal within the tolerance
equal_arrays = np.all(difference < tolerance, axis=(1, 2, 3))

# Use the mask to filter out sequences where input equals output
x = xx[~equal_arrays]
y = yy[~equal_arrays]

# Optional: Print the number of sequences removed
num_removed = np.sum(equal_arrays)
print(f"Number of sequences where input equals output within tolerance: {num_removed}")


#%%
# ntrain =int(0.9*y.shape[0])
# ntest = int(0.1*y.shape[0])
# train_a = x[ntest:ntrain+ntest, :, :, :]
# test_a = x[:ntest,:,:,:]

# train_u = y[ntest:ntrain+ntest, :, :, :]
# test_u = y[:ntest,:,:,:]
# np.save('test_a', test_a)
# np.save('test_u', test_u)

from sklearn.model_selection import train_test_split

train_a, test_a, train_u, test_u = train_test_split(
    x, y, test_size=0.1, random_state=42
)

#%% parameters for training 
batch_size = 20
batch_size2 = batch_size
modes = 20
width = 40
epochs = 300
learning_rate = 0.001
scheduler_step = 100
scheduler_gamma = 0.5

S = 64
T_in = 10
T = 10
step = 1

#%%
train_a = torch.from_numpy(train_a)
test_a = torch.from_numpy(test_a)

train_u = torch.from_numpy(train_u)
test_u = torch.from_numpy(test_u)

#%%

train_a = train_a.permute(0,2, 3, 1) 
test_a = test_a.permute(0,2, 3, 1) 

train_u = train_u.permute(0,2, 3, 1) 
test_u = test_u.permute(0,2, 3, 1) 
#%%
# train_a = train_a.half()
# test_a = test_a.half()
# train_u = train_u.half()
# test_u = test_u.half()

# Save test inputs and outputs
np.save('test_a.npy', test_a.cpu().numpy())
np.save('test_u.npy', test_u.cpu().numpy())


#%%
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)
#%%
scaler = torch.cuda.amp.GradScaler()

size_x = 64
size_y = 64
class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (self.in_channels * self.out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()



        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        lin = T+2
        self.fc0 = nn.Linear(lin, self.width)
        
        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic
        ################################    
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
        #x = self.bn0(x)
        x = F.gelu(x)


        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
#%%
model = FNO2d (modes, modes, width).cuda()
print(count_params(model))

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
# myloss = torch.nn.L1Loss( reduction = 'mean')
# myloss = torch.nn.L1Loss (size_average=None, reduce=None, reduction='mean')
myloss = LpLoss(size_average = False)

los_train = []
los_test = []

#%%
print('Begining of training')

print(modes)
ntrain = len(train_loader.dataset)
ntest = len(test_loader.dataset)
from timeit import default_timer
total_start_time = default_timer()

for ep in tqdm (range(epochs)):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    for xx, yy in train_loader:
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)
        current_batch_size = xx.shape[0]
        for t in range(0, T, step):
            y = yy[..., t:t + step]
            im = model(xx)
            iii = im.reshape(current_batch_size, -1)
            loss += myloss(
                im.reshape(current_batch_size, -1),
                y.reshape(current_batch_size, -1)
            )

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)
            iman = xx[..., step:]    
            xx = torch.cat((xx[..., step:], im), dim=-1)

        train_l2_step += loss.item()
        l2_full = myloss(pred.reshape(current_batch_size, -1), yy.reshape(current_batch_size, -1))
        train_l2_full += l2_full.item()
        #los_train.append(train_l2_full)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

    test_l2_step = 0.0
    test_l2_full = 0.0
    with torch.no_grad():
        model.eval()
        for xx, yy in test_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)
            current_batch_size = xx.shape[0]
            for t in range(0, T, step):
                y = yy[..., t:t + step]
                im = model(xx)
                loss += myloss(im.reshape(current_batch_size, -1), y.reshape(current_batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., step:], im), dim=-1)

            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(current_batch_size, -1), yy.reshape(current_batch_size, -1)).item()
               
    t2 = default_timer()
    scheduler.step()

    print( f' train: {train_l2_step / ntrain / (T / step), train_l2_full / ntrain}, test: {test_l2_step / ntest / (T / step), test_l2_full / ntest}')
    los_train.append(train_l2_full / ntrain)
    los_test.append(test_l2_full / ntest) 

total_end_time = default_timer()

# Calculate total time for the whole training process
total_time = total_end_time - total_start_time

# Print the total time for the full process
print(f"Total training process time: {total_time:.2f} seconds")
#%%
#import os
#path = os.getcwd()
#path_model1 = os.path.join(path, "FNO_model_Lploss_308")
## path_model2 = os.path.join(path, "FNO_model5_dict2")           
#torch.save(model, path_model1) 
# torch.save(model.state_dict(), path_model2)
#%%
#import pickle
#
#with open('loss_test', 'wb') as fp:
#    pickle.dump(los_test, fp)
#    
#with open('loss_train', 'wb') as fm:
#    pickle.dump(los_train, fm)
#%%
path = os.getcwd()
model_dir = os.path.join(path, 'saved_models')
os.makedirs(model_dir, exist_ok=True)
path_model1 = os.path.join(model_dir, "FNO_model.pth")

# Set model to evaluation mode before saving
model.eval()
torch.save(model.state_dict(), path_model1)

# Save loss values as CSV files
import csv

# Save training loss
with open('loss_train.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'Train Loss'])
    for epoch, loss in enumerate(los_train, 1):
        writer.writerow([epoch, loss])

# Save testing loss
with open('loss_test.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'Test Loss'])
    for epoch, loss in enumerate(los_test, 1):
        writer.writerow([epoch, loss])

# Optionally, save training metadata
training_info = {
    'epochs': epochs,
    'learning_rate': learning_rate,
    'batch_size': batch_size,
    'modes': modes,
    'width': width,
    'scheduler_step': scheduler_step,
    'scheduler_gamma': scheduler_gamma,
    'loss_train': los_train,
    'loss_test': los_test
}

with open('training_info.pkl', 'wb') as f:
    pickle.dump(training_info, f)


print("All done")