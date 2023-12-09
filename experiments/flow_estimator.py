#%%
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import torch.optim as optim
import normflows as nf
from typing import List


#%%
class FullyConnected(nn.Module):
    def __init__(self, hidden_sizes: List[int], dim_input: int, activation=nn.ReLU()):
        super().__init__()

        hidden_sizes = [dim_input] + hidden_sizes

        self.net = []

        for i in range(len(hidden_sizes) - 1):
            self.net.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.net.append(activation)

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


# class FlowModel(nn.Module):
#     def __init__(self, flows: List[str], dim_z: int, activation=torch.tanh):
#         super().__init__()
#
#         self.prior = MultivariateNormal(torch.zeros(dim_z), torch.eye(dim_z))
#         self.net = []
#
#         for i in range(len(flows)):
#             layer_class = eval(flows[i])
#             self.net.append(layer_class(dim_z, activation))
#
#         self.net = nn.Sequential(*self.net)
#
#         self.dim_z = dim_z
#
#     def forward(self, mu: torch.Tensor, log_sigma: torch.Tensor):
#         """
#         mu: tensor with shape (batch_size, D)
#         sigma: tensor with shape (batch_size, D)
#         """
#         sigma = torch.exp(log_sigma)
#         batch_size = mu.shape[0]
#         samples = self.prior.sample(torch.Size([batch_size]))
#         z = samples * sigma + mu
#
#         z0 = z.clone().detach()
#         log_prob_z0 = torch.sum(-0.5 * torch.log(torch.tensor(2 * torch.pi)) - log_sigma - 0.5 * ((z0 - mu) / sigma) ** 2, axis=1)
#
#         log_det = torch.zeros((batch_size,))
#
#         for layer in self.net:
#             z, ld = layer(z)
#             log_det += ld
#
#         log_prob_zk = torch.sum(-0.5 * (torch.log(torch.tensor(2 * torch.pi)) + z ** 2), axis=1)
#
#         return z, log_prob_z0, log_prob_zk, log_det

#%%
# Define 2D Gaussian base distribution
base = nf.distributions.base.DiagGaussian(1)

# Define list of flows
num_layers = 8
flows = []
for i in range(num_layers):
    # Neural network with two hidden layers having 64 units each
    # Last layer is initialized by zeros making training more stable
    param_map = nf.nets.MLP([1, 64, 64, 1], init_zeros=True)
    # Add flow layer
    flows.append(nf.flows.Planar(shape=[200, 1]))
    # flows.append(nf.flows.AffineCouplingBlock(param_map))
    # Swap dimensions
    # flows.append(nf.flows.Permute(1, mode='swap'))

# Construct flow model
model = nf.NormalizingFlow(base, flows)
target = nf.distributions.DiagGaussian(1)
# target = nf.distributions.Uniform(0, 1)

#%%
grid_size = 200
z = torch.linspace(-3, 3, grid_size).unsqueeze(1)

#%%
log_prob = target.log_prob(z).to('cpu')#.view(*xx.shape)

#%%
model.eval()
log_prob = model.log_prob(z).to('cpu')#.view(*xx.shape)
model.train()
prob = torch.exp(log_prob)
prob[torch.isnan(prob)] = 0

#%%
plt.figure()
plt.plot(z, log_prob.data.numpy())
# plt.gca().set_aspect('equal', 'box')
plt.show()


#%%
n_qubit = 2
n_seq = 10
n_batch = 4

s = torch.randn([n_batch, n_seq, n_qubit])

encoder = nn.GRU(
    input_size=2,  # number of qubits, nq
    hidden_size=32,  # encoder hidden size, H
    num_layers=1,
    batch_first=True
)
output, hn = encoder.forward(s)
encoding = hn.squeeze()  # output[:, -1, :]
print(encoding.shape)

fc = FullyConnected(input_size=32, hidden_size=16, output_size=2)
latent = fc(encoding)
print(latent.shape)

mu = latent[:, 0]
sigma = latent[:, 1]

print(mu, sigma)

#%%
