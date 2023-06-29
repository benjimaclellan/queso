import time
from typing import Sequence
import tqdm
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
import optax
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import seaborn as sns

from queso.io import IO

colors = sns.color_palette('deep', n_colors=8)


#%%
def bit_to_integer(a, endian='le'):
    if endian == 'le':
        k = 1 << jnp.arange(a.shape[-1] - 1, -1, -1)  # little-endian
    elif endian == 'be':
        k = 1 << jnp.arange(a.shape[-1] - 1, -1, -1)
    else:
        raise NotImplementedError
    s = jnp.einsum('ijk,k->ij', a, k)
    return s.type(jnp.float32).unsqueeze(2)


#%%
class BayesianNetwork(nn.Module):
    dims: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for dim in self.dims[:-1]:
            x = nn.relu(nn.Dense(dim)(x))
        x = nn.Dense(self.dims[-1])(x)
        # x = nn.activation.softmax(x, axis=-1)
        return x


#%%
io = IO(folder='bayesian-net')
save = False
show = True

n = 1
n_shots = 1000
n_phis = 200
n_output = 30  # number of output neurons (discretization of phase range)

dims = [4, 4, n_output]
n_steps = 5000
batch_phis = 32
batch_shots = 16
progress = True
lr = 1e-3

phi_range = (0, jnp.pi)
phis = jnp.linspace(phi_range[0], phi_range[1], n_phis)
choices = jnp.array([0, 1])


#%%
probs = jnp.array([jnp.cos(phis / 2) ** 2, jnp.sin(phis / 2) ** 2])


@jax.vmap
def sample(phi, key):
    probs = jnp.array([jnp.cos(phi / 2)**2, jnp.sin(phi / 2)**2])
    return jax.random.choice(key, choices, shape=(n_shots, 1), p=probs)


key = jax.random.PRNGKey(time.time_ns())
subkeys = jax.random.split(key, num=n_phis)
outcomes = sample(phis, subkeys)

#%%
dphi = (phi_range[1] - phi_range[0]) / (n_output - 1)

index = jnp.round(phis / dphi)
labels = jax.nn.one_hot(index, num_classes=n_output)
# print(index)
# print(labels.shape)

print(labels.sum(axis=0))

#%%
model = BayesianNetwork(dims)


#%%
x_init = outcomes[1:10, 1:10, :]
print(model.tabulate(jax.random.PRNGKey(0), x_init))


#%%
@jax.jit
def train_step(state, batch):
    x, labels = batch
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, x)
        loss = optax.softmax_cross_entropy(logits, labels).mean(axis=(0, 1))
        return loss
    loss_val_grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = loss_val_grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


#%%
def create_train_state(model, init_key, x, learning_rate):
    params = model.init(init_key, x)['params']
    tx = optax.adam(learning_rate=learning_rate)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state


init_key = jax.random.PRNGKey(time.time_ns())
state = create_train_state(model, init_key, x_init, learning_rate=lr)
# del init_key

#%%
for i in (pbar := tqdm.tqdm(range(n_steps), disable=(not progress), mininterval=0.333)):
    # generate batch
    key = jax.random.PRNGKey(time.time_ns())
    subkeys = jax.random.split(key, num=2)

    inds = jax.random.randint(subkeys[0], minval=0, maxval=n_phis, shape=(batch_phis,))

    x = outcomes[inds, :, :]
    idx = jax.random.randint(subkeys[1], minval=0, maxval=n_shots, shape=(batch_phis, batch_shots, 1))
    x = jnp.take_along_axis(x, idx, axis=1)

    #%%
    # x = outcomes[inds, jnds, :]
    y = jnp.repeat(jnp.expand_dims(labels[inds], 1), repeats=batch_shots, axis=1)
    batch = (x, y)

    state, loss = train_step(state, batch)
    if progress:
        pbar.set_description(f"Step {i} | FI: {loss:.10f}", refresh=False)

#%%
state_params = state.params
# model.apply(state_params, x_init)
pred = state.apply_fn({'params': state.params}, jnp.array([[0], [1]]))
pred = nn.activation.softmax(jnp.exp(pred), axis=-1)

fig, axs = plt.subplots(nrows=2, sharex=True)
ax = axs[0]
ax.plot(jnp.linspace(phi_range[0], phi_range[1], pred.shape[1]), pred[0, :], ls='-', color=colors[0], label='Pr(0)')
ax.plot(jnp.linspace(phi_range[0], phi_range[1], pred.shape[1]), pred[1, :], ls='-', color=colors[1], label='Pr(1)')
ax.legend()

ax = axs[1]
ax.plot(jnp.linspace(phi_range[0], phi_range[1], probs.shape[1]), probs[0], ls='--', color=colors[0], label='Pr(0) Truth')
ax.plot(jnp.linspace(phi_range[0], phi_range[1], probs.shape[1]), probs[1], ls='--', color=colors[1], label='Pr(1) Truth')
ax.set(xlabel='Phi Prediction', ylabel="Probability")

if show:
    plt.show()
if save:
    io.save_figure(fig, filename="probs.png")

#%% 
fig, axs = plt.subplots(nrows=3)
colors = sns.color_palette('crest', n_colors=10)
x = jnp.linspace(*phi_range, n_output)

for i, m in enumerate([1, 10, 30]):
    ax = axs[i]
    for j, k in enumerate(range(0, n_phis, 20)):
        phi = phis[k]
        shots = outcomes[k, :m]
        pred = state.apply_fn({'params': state.params}, shots)
        pred = nn.activation.softmax(pred, axis=-1)
        # pred = nn.activation.softmax(jnp.exp(pred), axis=-1)
        pred = pred.prod(axis=0)
        pred = pred / jnp.max(pred)

        ax.plot(x, pred, color=colors[j])
        ax.axvline(phi, color=colors[j], ls='--', alpha=0.4)

if show:
    plt.show()
if save:
    io.save_figure(fig, filename="sequence.png")
