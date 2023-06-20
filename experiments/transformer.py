# %%
import time
import datetime
import numpy as np
import h5py
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable

from queso.io import IO
from queso.estimators.transformer import Encoder
from queso.estimators.data import SensorDataset, SensorSampler
from queso.utils import shots_to_counts

#%%
n = 6
k = 6
io = IO(folder=f"2023-06-07_nn-estimator-n{n}-k{k}")
hf = h5py.File(io.path.joinpath("circ.h5"), "r")

# %%
device = 'cuda'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

progress = True
save = True
plot = True

n_epoch = 1
n_batch = 512

d_model = n
d_ff = 10
dropout = 0.1
num_heads = 1
n_layers = 8

n_steps = 10000
lr = 1e-3

#%%
# cutoff = 160
inds = torch.arange(0, 20, 1)
shots = torch.tensor(np.array(hf.get("shots")), dtype=torch.float32)
phis = torch.tensor(np.array(hf.get("phis")), dtype=torch.float32).unsqueeze(dim=1)
shots = shots[inds, :, :]
phis = phis[inds, :]

shots = shots.to(device)
phis = phis.to(device)

#%% io for saving plots at the end
io = IO(folder="transformer-graham", include_date=True, include_time=True)

#%%
n_phis = shots.shape[0]
n_shots = shots.shape[1]
n = shots.shape[2]

#%%
encoder = Encoder(d_model=d_model, n_layers=n_layers, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
encoder.to(device)

# encoder
# torch.nn.init.uniform_(encoder.weight)
# nn.init.xavier_uniform_(nn.Linear(2, 2))


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


#%%
count_parameters(encoder)

#%%
dataset = SensorDataset(shots, phis)
sampler = SensorSampler(dataset, replacement=True, n_samples=n_batch)

# %%
train_loader = data.DataLoader(dataset, sampler=sampler, batch_size=None)

#%%
x, y = next(iter(train_loader))
print(x.shape)
print(encoder.encoder_layer(x).shape)
#%%
for batch_ind in range(1):
    x, y = next(iter(train_loader))
    pred = encoder(x)
    print(batch_ind, x.shape, y.shape, pred.squeeze())

#%%
pred = encoder(x)
print(pred.shape)

# %%
step = 0
t0 = time.time()
start = datetime.datetime.now()
losses = []

#%%
criterion = nn.MSELoss()
optimizer = optim.Adam(encoder.parameters(), lr=lr) #, betas=(0.9, 0.98), eps=1e-9)

encoder.train()

#%%
for step in (pbar := tqdm.tqdm(range(n_steps), disable=(not progress))):
    x, y = next(iter(train_loader))
    pred = encoder(x)

    optimizer.zero_grad()
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    # print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
    if progress:
        pbar.set_description(f"MSE: {loss.item():.10f}")

#%%
fig, ax = plt.subplots()
ax.plot(losses)
ax.set(xlabel='Iteration', ylabel='MSE Loss', yscale='log')
if save:
    io.save_figure(fig, filename="loss.png")
if plot:
    fig.show()

#%%
phis_est = encoder(shots[:, :100, :])
# print(torch.stack([phis, phis_est], dim=2))

#%%
fig, ax = plt.subplots()
ax.plot(phis.detach().cpu().numpy(), label="Truth")
ax.plot(phis_est.detach().cpu().numpy(), label='Estimate')
ax.legend()
if save: 
    io.save_figure(fig, filename="estimate.png")
if plot:
    fig.show()


#%%

# counts = shots_to_counts(shots[:, :100, :], phis)
# sns.heatmap(counts)
# # sns.heatmap(shots.mean(dim=1).numpy())
# plt.show()

#%%
