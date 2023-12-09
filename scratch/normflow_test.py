#%%
import torch
from torch import nn
import normflows as nf
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import seaborn as sns

import einops


#%%
# Move model on GPU if available
enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')

#%%
# target = nf.distributions.GaussianMixture(n_modes=2, dim=1, loc=[[-1], [1]], scale=[[0.3], [0.3]])
target = nf.distributions.GaussianMixture(n_modes=1, dim=1, loc=[[1]], scale=[[0.3]])
y = target.sample(1000)
plt.figure()
plt.hist(y[:, 0].detach(), bins=100)
plt.show()

#%%
# Plot target distribution
grid_size = 200
zz = torch.linspace(-3, 3, grid_size)
zz = zz.view(-1, 1)
zz = zz.to(device)

log_prob = target.log_prob(zz).to('cpu')  #.view(*xx.shape)
prob = torch.exp(log_prob)
prob[torch.isnan(prob)] = 0

plt.figure()
plt.plot(zz, prob.data.numpy())
plt.show()

#%%
base = nf.distributions.base.DiagGaussian(1, trainable=False)

# Define list of flows
num_layers = 12
flows = []
for i in range(num_layers):
    # param_map = nf.nets.MLP([1, 64, 64, 1], init_zeros=True)
    # flows.append(nf.flows.AffineCouplingBlock(param_map))
    flows.append(nf.flows.AffineConstFlow(shape=[1]))

# Construct flow model
model = nf.NormalizingFlow(base, flows)
model = model.to(device)

#%%
# Plot initial flow distribution
model.eval()
log_prob = model.log_prob(zz).to('cpu')  #.view(*xx.shape)
model.train()
prob = torch.exp(log_prob)
prob[torch.isnan(prob)] = 0

#%%
n_samples = 1000
x = base.sample(n_samples).detach()
y = target.sample(n_samples).detach()
z, _ = model.sample(n_samples)
z = z.detach()
plt.figure()
plt.hist(x[:, 0], bins=100, alpha=0.3, label='base')
plt.hist(y[:, 0], bins=100, alpha=0.3, label='target')
plt.hist(z[:, 0], bins=100, alpha=0.3, label='flow')
plt.legend()
plt.show()

#%% Train model
max_iter = 1000
num_samples = 2 ** 9
show_iter = 500

loss_hist = np.array([])
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
for it in tqdm(range(max_iter)):
    optimizer.zero_grad()

    # Get training samples
    x = target.sample(num_samples).to(device)

    # Compute loss
    loss = model.forward_kld(x)

    # Do backprop and optimizer step
    if ~(torch.isnan(loss) | torch.isinf(loss)):
        loss.backward()
        optimizer.step()

    # Log loss
    loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())

    # # Plot learned distribution
    # if (it + 1) % show_iter == 0:
    #     model.eval()
    #     log_prob = model.log_prob(zz)
    #     model.train()
    #     prob = torch.exp(log_prob.to('cpu').view(*xx.shape))
    #     prob[torch.isnan(prob)] = 0

#%% Plot loss
plt.figure(figsize=(10, 10))
plt.plot(loss_hist, label='loss')
plt.legend()
plt.show()
