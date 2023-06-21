# %%
import time
import datetime
import numpy as np
import h5py
import pandas as pd
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable

from queso.io import IO
from queso.estimators.data import SensorDataset, SensorSampler
from queso.utils import shots_to_counts, count_parameters


#%%
def bit_to_integer(a, endian='le'):
    if endian == 'le':
        k = 1 << torch.arange(a.shape[-1] - 1, -1, -1)  # little-endian
    elif endian == 'be':
        k = 1 << torch.arange(a.shape[-1] - 1, -1, -1)
    else:
        raise NotImplementedError
    s = torch.einsum('ijk,k->ij', a, k)
    return s.type(torch.float32).unsqueeze(2)


#%%
class BayesianEstimator(nn.Module):
    def __init__(self, dims: list = None):
        super().__init__()
        if dims is None:
            dims = [1, 10, 10, 100]
        assert dims[0] == 1

        net = []
        for i in range(1, len(dims)):
            net.append(nn.Linear(dims[i-1], dims[i]))
            # net.append(nn.Sigmoid())
            net.append(nn.GELU())
        net.append(nn.Softmax(dim=-1))
        self.net = nn.Sequential(*net)

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x


#%%
n = 1
n_shots = 2000
n_phis = 40
n_output = 30  # number of output neurons (discretization of phase range)
lr = 1e-3

phis = torch.linspace(0, torch.pi, n_phis, dtype=torch.float64)
outcomes = torch.zeros([n_phis, n_shots])
for i, phi in enumerate(phis):
    pr0 = np.cos(phi / 2)**2
    pr1 = np.sin(phi / 2)**2
    outcomes[i, :] = torch.tensor(np.random.choice([0, 1], size=n_shots, p=[pr0, pr1]))

outcomes = outcomes.unsqueeze(2)

#%%
# n = 2
# k = 2
# io = IO(folder=f"2023-06-07_nn-estimator-n{n}-k{k}")
# hf = h5py.File(io.path.joinpath("circ.h5"), "r")
#
# cutoff = 160
# shots = torch.tensor(np.array(hf.get("shots")), dtype=torch.int64)[:cutoff]
# phis = torch.tensor(np.array(hf.get("phis")), dtype=torch.float32)[:cutoff]


# outcomes = bit_to_integer(shots)
# print(outcomes)
# print(outcomes.shape, outcomes.type())

#%%
D = [0, phis[-1]]  # range of phase values in training set
dphi = (D[1] - D[0]) / (n_output - 1)

index = torch.floor(phis / dphi).type(torch.int64)
print(index)
labels = nn.functional.one_hot(index, num_classes=n_output).type(torch.FloatTensor)
print(labels.shape)

#%%
model = BayesianEstimator(dims=[1, 20, 20, n_output])

# pred = model(outcomes[:, 0, :])
# print(pred.shape)
# print(labels.shape)

count_parameters(model)

#%%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

model.train()

#%%
n_steps = 50000
progress = True
losses = []
for step in (pbar := tqdm.tqdm(range(n_steps), disable=(not progress))):
    inds = torch.randint(0, labels.shape[0], size=(128,))

    x = outcomes[inds, step % outcomes.shape[1], :]
    y = labels[inds]

    pred = model(x)

    optimizer.zero_grad()
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if progress:
        pbar.set_description(f"CE Loss: {loss.item():.10f}")

#%%
plt.plot(losses)
plt.show()

#%%
hist = torch.zeros([n_phis, 2])
for i, phi in enumerate(phis):
    hist[i, 0] = (outcomes[i, :, 0] == 0).sum()
    hist[i, 1] = (outcomes[i, :, 0] == 1).sum()

#%%
sns.heatmap(hist)
plt.show()

#%%
fig, ax = plt.subplots()
m_pred0 = model(torch.tensor([0]).type(outcomes.type()))
m_pred1 = model(torch.tensor([1]).type(outcomes.type()))
m = torch.stack([m_pred0, m_pred1], dim=1).detach()
sns.heatmap(m)
plt.show()

#%%