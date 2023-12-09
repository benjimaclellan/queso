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
class Dense(nn.Module):
    def __init__(self, layer_widths, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layers = []
        for i in range(len(layer_widths) - 1):
            layers.append(nn.Linear(layer_widths[i], layer_widths[i + 1]))
            if i < len(layer_widths) - 2:
                layers.append(nn.LeakyReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Estimator(nn.Module):
    def __init__(self, encoder: nn.Module, base, flow: nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.base = base
        self.flow = flow

    def forward(self, x):
        latent = self.encoder(x)

        # sample base dist.
        eps = base.sample(1000).squeeze()
        z = latent[:, 0] + torch.exp(latent[:, 1]) * eps

        return self.model(x)


#%%
# Move model on GPU if available
enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')


#%%
n_phases = 50
n_seq = 100
n_qubits = 1

#%%
encoder = Dense([n_qubits, 4, 4, 4, 2])


#%%
def sample(phase: float):
    p = np.array([np.cos(phase - np.pi/4) ** 2, np.sin(phase - np.pi/4) ** 2])
    p = p / np.sum(p)
    # print(p, sum(p))
    bit = np.random.choice(
        [0, 1],
        (n_seq, n_qubits),
        p=p
    )
    # s = np.tile(logical, [n_qubits, 1])
    return bit


labels = np.arange(n_phases,)
phases = np.linspace(-np.pi/4, np.pi/4, n_phases)
shots = np.array(list(map(lambda phase: sample(phase=phase), phases)))
print(shots.shape)
# print(shots)

#%% prepare data in the correct shapes
shots_r = torch.Tensor(einops.rearrange(shots, "b seq q -> (b seq) q"))
phases_r = torch.Tensor(einops.repeat(phases, "b -> (b seq) 1", seq=n_seq))

#%%
out = encoder(shots_r)

#%%
grid_size = 200
zz = torch.linspace(-3, 3, grid_size)
zz = zz.view(-1, 1)
zz = zz.to(device)

#%%
base = nf.distributions.base.DiagGaussian(1, trainable=False)

# Define list of flows
num_layers = 4
flows = []
for i in range(num_layers):
    flows.append(nf.flows.AffineConstFlow(shape=[1]))
    # flows.append(nf.flows.Planar(shape=[1], act='leaky_relu'))
    # flows.append(nf.flows.Radial(shape=[1], z_0=None))
    # flows.append(nf.flows.Squeeze())
    nf.flows.CCAffineConst(shape=[1], num_classes=1)

# Construct flow model
flow = nf.NormalizingFlow(base, flows)
flow = flow.to(device)

#%%
# Plot initial flow distribution
flow.eval()
log_prob = flow.log_prob(zz).to('cpu')
flow.train()
prob = torch.exp(log_prob)
prob[torch.isnan(prob)] = 0

#%%
n_samples = 1000
x = base.sample(n_samples).detach()
z, _ = flow.sample(n_samples)
z = z.detach()
plt.figure()
plt.hist(x[:, 0], bins=100, alpha=0.3, label='base')
plt.hist(z[:, 0], bins=100, alpha=0.3, label='flow')
plt.legend()
plt.show()

#%% Train model
max_iter = 10000

loss_hist = np.array([])
parameters = list(encoder.parameters()) + list(flow.parameters())
optimizer = torch.optim.Adam(parameters, lr=5e-4, weight_decay=1e-5)

#%%
# for it in tqdm(range(max_iter)):
progress = True
pbar = tqdm(total=max_iter, disable=(not progress), mininterval=0.1)
for i in range(max_iter):
    optimizer.zero_grad()

    # forward pass to get latent vars
    latent = encoder(shots_r)

    # sample base dist.
    eps = base.sample(phases_r.shape[0])
    z0 = latent[:, 0, None] + torch.exp(latent[:, 1, None]) * eps
    z = flow.forward(z0)

    loss = einops.reduce(torch.pow(phases_r - z, 2), "b f -> ", "mean")

    # Do backprop and optimizer step
    if ~(torch.isnan(loss) | torch.isinf(loss)):
        loss.backward()
        optimizer.step()

    # Log loss
    loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())

    if progress:
        pbar.update()
        pbar.set_description(f"Step {i} | Loss: {loss:.10f}", refresh=False)

#%% Plot loss
plt.figure(figsize=(10, 10))
plt.plot(loss_hist, label='loss')
plt.legend()
plt.show()

#%%
plt.figure()
plt.scatter(phases_r.squeeze(), z.detach().squeeze(), alpha=0.1)
plt.show()

#%%

