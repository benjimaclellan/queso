#%%
import torch
import flowtorch.bijectors as bij
import flowtorch.distributions as dist
import matplotlib.pyplot as plt

#%%
dim_latent = 1
base_dist = torch.distributions.Independent(
  torch.distributions.Normal(torch.zeros(dim_latent), torch.ones(dim_latent)),
  1
)


# target_dist = torch.distributions.Independent(
#   torch.distributions.Normal(torch.zeros(dim_latent)+5, torch.ones(dim_latent)*0.1),
#   1
# )

# Create two normal distributions with the specified parameters
# dist1 = torch.distributions.Normal(mean1, std1)
# dist2 = torch.distributions.Normal(mean2, std2)

# Combine the two distributions into a bimodal distribution
target_dist = torch.distributions.MixtureSameFamily(
    torch.distributions.Categorical(torch.tensor([[0.5, 0.5]])),
    torch.distributions.Normal(torch.tensor([[-4.5, 4.5]]), torch.tensor([[3, 3]]))
)


#%%
# bijectors = bij.AffineAutoregressive()
# bijectors = bij.Affine()
# bijectors = bij.Spline()
bijectors = bij.SplineAutoregressive()

# Instantiate transformed distribution and parameters
flow = dist.Flow(base_dist, bijectors)

#%%
# Training loop
opt = torch.optim.Adam(flow.parameters(), lr=2e-3)
for idx in range(3001):
    opt.zero_grad()

    # Minimize KL(p || q)
    y = target_dist.sample((1000,))
    loss = -flow.log_prob(y).mean()

    if idx % 500 == 0:
        print('epoch', idx, 'loss', loss)

    loss.backward()
    opt.step()

#%%
plt.figure()
x = base_dist.sample((1000,))
y = target_dist.sample((1000,))
z = flow.sample((1000,))
if dim_latent == 1:
    plt.scatter(x[:, 0], 1 * torch.ones(1000), label='base')
    plt.scatter(y[:, 0], 2 * torch.ones(1000), label='target')
    # plt.scatter(y[:, 0], 2 * torch.ones(1000), label='target')
    plt.scatter(z[:, 0], 3 * torch.ones(1000), label='flow')

elif dim_latent == 2:
    plt.scatter(x[:, 0], x[:, 1], label='base')
    plt.scatter(y[:, 0], y[:, 1], label='target')
    plt.scatter(z[:, 0], z[:, 1], label='flow')

plt.legend()
plt.show()

#%%
plt.figure()
plt.hist(x[:, 0], bins=100, alpha=0.3, label='base')
plt.hist(y[:, 0], bins=100, alpha=0.3, label='target')
plt.hist(z[:, 0], bins=100, alpha=0.3, label='flow')
plt.legend()
plt.show()

#%%