# %%
import torch
import normflows as nf
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import einops

from queso.estimators.flow.models import RNN, Flow

# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_flash_sdp(False)
# torch.backends.cuda.enable_math_sdp(True)

# %%
# Move model on GPU if available
enable_cuda = True
# device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
# device = 'cpu'
device = "cuda"

# %%
n_phases = 20
n_seq = 10
n_shots = 5000
n_qubits = 6
n_train_seq = 50


# %%
def sample(phase: float):
    p = np.array(
        [np.cos(n_qubits * phase) ** 2, np.sin(n_qubits * phase) ** 2]
    )  # GHZ state
    p = p / np.sum(p)
    bit = np.random.choice([0, 1], n_shots, p=p)
    s = einops.repeat(bit, "s -> s q", q=n_qubits)
    return s


labels = np.arange(
    n_phases,
)
phases = torch.linspace(0, np.pi / n_qubits / 2, n_phases)
shots = torch.Tensor(np.array(list(map(lambda phase: sample(phase=phase), phases))))
print(shots.shape)

# %%
# encoder = Dense([n_qubits, 232, 232, 2]).to(device)
encoder = RNN(dim_input=n_qubits, dim_hidden=32, dim_output=2, num_layers=4).to(device)

# encoder_layer = nn.TransformerEncoderLayer(
#     d_model=n_qubits,
#     nhead=1,
#     dim_feedforward=64,
#     batch_first=True,
#     norm_first=False,
# ).to(device)
# encoder = nn.TransformerEncoder(
#     encoder_layer,
#     num_layers=6,
#     enable_nested_tensor=False,
# ).to(device)
# src = torch.rand(10, 32, 512)
# out = transformer_encoder(src)


# %% prepare data in the correct shapes
def sequence(shots: torch.Tensor, phases: torch.Tensor, n_seq: int):
    batch_phase = n_shots // n_seq
    trunc = n_shots - n_shots % n_seq

    shots_r = shots[:, torch.randperm(shots.shape[1]), :]
    shots_r = shots_r[:, :trunc, :]
    shots_r = torch.Tensor(
        einops.rearrange(shots_r, "b (bp seq) q -> (b bp) seq q", bp=batch_phase)
    ).to(device)

    phases_r = torch.Tensor(einops.repeat(phases, "b -> (b bp) 1", bp=batch_phase)).to(
        device
    )
    return shots_r, phases_r


shots_r, phases_r = sequence(shots, phases, n_seq=n_seq)

# %%
# for i in range(phases_r.shape[0]):
#     print(phases_r[i, 0], shots_r[i, :, :])

# %%
out = encoder(shots_r)

# %%
base = nf.distributions.base.DiagGaussian(1, trainable=False).to(device)

flow = Flow(base=base, num_layers=4).to(device)
flow.sample(10)

# %%
# grid_size = 200
# zz = torch.linspace(-3, 3, grid_size)
# zz = zz.view(-1, 1)
# zz = zz.to(device)
#
# # Plot initial flow distribution
# flow.eval()
# log_prob = flow.log_prob(zz).to('cpu')
# flow.train()
# prob = torch.exp(log_prob)
# prob[torch.isnan(prob)] = 0
#
# n_samples = 1000
# x = base.sample(n_samples).detach()
# z, _ = flow.sample(n_samples)
# z = z.detach()
# plt.figure()
# plt.hist(x[:, 0].cpu(), bins=100, alpha=0.3, label='base')
# plt.hist(z[:, 0].cpu(), bins=100, alpha=0.3, label='flow')
# plt.legend()
# plt.show()

# %% Train model
max_iter = 500

loss_hist = np.array([])
parameters = list(encoder.parameters()) + list(flow.parameters())
optimizer = torch.optim.Adam(parameters, lr=1e-4, weight_decay=1e-3)

# %%
progress = True

for n_seq in (n_train_seq,):
    pbar = tqdm(total=max_iter, disable=(not progress), mininterval=0.1)
    shots_r, phases_r = sequence(shots, phases, n_seq=n_seq)

    for i in range(max_iter):
        optimizer.zero_grad()

        # forward pass to get latent vars
        latent = encoder(shots_r)

        # sample base dist.
        eps = base.sample(phases_r.shape[0])
        # z0 = latent[:, 0, None] + torch.exp(latent[:, 1, None]) * eps
        z0 = latent[:, 0, None] + latent[:, 1, None] * eps
        z = flow.forward(z0)

        loss = einops.reduce(torch.pow(phases_r - z, 2), "b f -> ", "mean")
        # loss = einops.reduce(torch.abs(phases_r - z), "b f -> ", "mean")

        # Do backprop and optimizer step
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()

        # Log loss
        loss_hist = np.append(loss_hist, loss.to("cpu").data.cpu().numpy())

        if progress:
            pbar.update()
            pbar.set_description(
                f"Step {i} | Loss: {loss:.10f} | sequence length {n_seq}", refresh=False
            )


# %%
flow.eval()
encoder.eval()

# %% Plot loss
plt.figure(figsize=(10, 10))
plt.plot(loss_hist, label="loss")
plt.legend()
plt.show()

# %%
fig, axs = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=True)

for i, n_seq in enumerate([2, 20, 50, 100]):
    ax = axs[i]
    shots_r, phases_r = sequence(shots, phases, n_seq=n_seq)

    latent = encoder(shots_r)

    eps = base.sample(phases_r.shape[0])
    # z0 = latent[:, 0, None] + torch.exp(latent[:, 1, None]) * eps
    z0 = latent[:, 0, None] + latent[:, 1, None] * eps
    z = flow.forward(z0)

    ax.scatter(
        phases_r.cpu().squeeze(),
        z.cpu().detach().squeeze(),
        alpha=0.1,
        label=f"infer, sequence length $m={n_seq}$",
    )
    ax.set(ylim=[0, torch.pi / 2 / n_qubits])
    ax.legend()
axs[0].set(title=f"train, sequence length, $m={n_train_seq}$")
axs[-1].set(
    xlabel=r"Phase (ground truth), $\varphi$",
    ylabel=r"Estimated phase, $\bar{\varphi}$",
)
fig.show()

fig.savefig("rnn_nf.png")


# %%
fig, axs = plt.subplots(nrows=4, ncols=2, sharex=True, sharey=True)

for i, n_seq in enumerate([2, 20, 50, 100]):
    ax = axs[i]
    shots_r, phases_r = sequence(shots, phases, n_seq=n_seq)

    latent = encoder(shots_r)

    axs[i, 0].scatter(
        phases_r.cpu().squeeze(), latent[:, 0].cpu().detach().squeeze(), alpha=0.1
    )
    axs[i, 1].scatter(
        phases_r.cpu().squeeze(), latent[:, 1].cpu().detach().squeeze(), alpha=0.1
    )
    # ax.set(ylim=[0, torch.pi/2/n_qubits])

axs[-1, 0].set(
    xlabel=r"Phase (ground truth), $\varphi$", ylabel=r"Latent var., $\lambda$"
)
axs[-1, 1].set(
    xlabel=r"Phase (ground truth), $\varphi$", ylabel=r"Latent var., $\sigma$"
)
fig.show()


# %% bias, variance at one phase value
flow.eval()

n_seq = 100
shots_t, phases_t = sequence(
    shots[None, n_phases // 2, :, :], phases[None, n_phases // 2], n_seq=n_seq
)
latent = encoder(shots_t)
# eps = base.sample(phases_t.shape[0])
grid_size = 64
# eps = torch.linspace(-1, 1, grid_size).to(device)
eps = torch.linspace(-1, 1, grid_size).to(device)
# z0 = latent[:, 0, None] + torch.exp(latent[:, 1, None]) * eps
z0 = latent[:, 0, None, None] + latent[:, 1, None, None] * eps[None, :, None]


z = flow.forward(z0)
log_prob = flow.log_prob(z0).to("cpu")
prob = torch.exp(log_prob)

# %%
fig, ax = plt.subplots()
for i in range(prob.shape[0]):
    ax.plot(eps.cpu(), prob[i, :].cpu().detach())
# plt.hist(x[:, 0].cpu(), bins=100, alpha=0.3, label='base')
# plt.hist(z[:, 0].cpu(), bins=100, alpha=0.3, label='flow')
ax.legend()
fig.show()
