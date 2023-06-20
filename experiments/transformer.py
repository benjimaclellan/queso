# %%
import time
import datetime
import numpy as np
import h5py

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

# %%
n = 6
k = 6
io = IO(folder=f"2023-06-07_nn-estimator-n{n}-k{k}")
hf = h5py.File(io.path.joinpath("circ.h5"), "r")

# cutoff = 160
inds = torch.arange(0, 20, 1)
shots = torch.tensor(np.array(hf.get("shots")), dtype=torch.float32)
phis = torch.tensor(np.array(hf.get("phis")), dtype=torch.float32).unsqueeze(dim=1)
shots = shots[inds, :, :]
phis = phis[inds, :]

#%%
n_phis = shots.shape[0]
n_shots = shots.shape[1]
n = shots.shape[2]
n_batch = 1024

# %%
d_model = n
d_ff = 16
dropout = 0.1

encoder = Encoder(d_model=d_model, n_layers=8, num_heads=2, d_ff=d_ff, dropout=0.0)

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
# pred = encoder_layer(x)
pred = encoder(x)
print(pred.shape)

# %%
n_epoch = 1
progress = True
device = "cpu"

# %%
step = 0
t0 = time.time()
start = datetime.datetime.now()

losses = []

#%%
criterion = nn.MSELoss()
optimizer = optim.Adam(encoder.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
# optimizer = optim.Adagrad(encoder.parameters(), lr=0.001)

encoder.train()

#%%
for epoch in range(5):
    x, y = next(iter(train_loader))
    pred = encoder(x)

    optimizer.zero_grad()
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

#%%
fig, ax = plt.subplots()
ax.plot(losses)
ax.set(xlabel='Iteration', ylabel='MSE Loss')
plt.show()

#%%
phis_est = encoder(shots[:, :100, :])
print(torch.stack([phis, phis_est], dim=2))

#%%
fig, ax = plt.subplots()
ax.plot(phis.numpy(), label="Truth")
ax.plot(phis_est.detach().numpy(), label='Estimate')
ax.legend()
plt.show()


#%%

# counts = shots_to_counts(shots[:, :100, :], phis)
# sns.heatmap(counts)
# # sns.heatmap(shots.mean(dim=1).numpy())
# plt.show()

#%%
