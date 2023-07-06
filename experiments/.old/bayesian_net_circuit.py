import copy
import time
from functools import partial
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
import seaborn as sns
import tensorcircuit as tc

from queso.io import IO
from queso.utils import bit_to_integer

backend = tc.set_backend("jax")
tc.set_dtype("complex128")
tc.set_contractor("auto")

colors = sns.color_palette('deep', n_colors=8)



#%%
class BayesianNetwork(nn.Module):
    nn_dims: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for dim in self.dims[:-1]:
            x = nn.relu(nn.Dense(dim)(x))
        x = nn.Dense(self.dims[-1])(x)
        # x = nn.activation.softmax(x, axis=-1)
        return x


#%%
def circuit(n, phi):
    c = tc.Circuit(n)
    c.h(0)
    for i in range(1, n):
        c.cnot(0, i)
    for i in range(n):
        c.rz(i, theta=phi)
    for i in range(n):
        c.h(i)
    return c


@partial(jax.jit, static_argnums=(0,))
def _sample(n, phi, key):
    backend.set_random_state(key)
    c = circuit(n, phi)
    return c.measure(*list(range(n)))[0]


@partial(jax.jit, static_argnums=(0,))
def probability(n, phi):
    c = circuit(n, phi)
    return c.probability()


def sample(n, phi, key=None, n_shots=100):
    if key is None:
        key = jax.random.PRNGKey(time.time_ns())
    keys = jax.random.split(key, n_shots)
    shots = jnp.array([_sample(n, phi, key) for key in keys]).astype(
        "int8"
    )
    return shots


def sample_over_phases(n, phis, n_shots, key=None):
    if key is None:
        key = jax.random.PRNGKey(time.time_ns())
    keys = jax.random.split(key, phis.shape[0])
    data = jnp.stack(
        [
            sample(n, phi, key=key, n_shots=n_shots)
            for (phi, key) in zip(phis, keys)
        ],
        axis=0,
    )
    probs = jnp.stack([probability(n, phi) for phi in phis], axis=0)
    return data, probs


# todo: normalize posterior distributions
# todo: create torch dataset object to have non-uniform number of samples
# todo: test variance and bias of estimator
# todo: performance when training phase is erroneous
# todo:


#%%
io = IO(folder='bayesian-net')
save = False
show = True
verbose = False

# input_type = 'value'
input_type = 'bits'

n = 2
n_shots = 200
n_phis = 50
n_grid = 50  # number of output neurons (discretization of phase range)

nn_dims = [16, 16, n_grid]
n_steps = 50000
batch_phis = 32
batch_shots = 1
progress = True
lr = 1e-3

phi_range = (0, jnp.pi)
phis = jnp.linspace(*phi_range, n_phis, endpoint=False)
delta_phi = (phi_range[1] - phi_range[0]) / (n_grid - 1)
index = jnp.floor(n_grid * (phis / (phi_range[1] - phi_range[0])))
labels = jax.nn.one_hot(index, num_classes=n_grid)
print(index)
print(labels.sum(axis=0))


samples, probs = sample_over_phases(n, phis, n_shots=n_shots)
full_samples = copy.copy(samples)
samples = samples[:n_phis//2, :, :]

if input_type == "value":
    outcomes = bit_to_integer(samples)  # data into network is real number (i.e., 0, 1, 2, 3, 4, ...)
elif input_type == "bits":
    outcomes = samples
else:
    raise ValueError

if verbose:
    print(probs)
    print(samples.squeeze().sum(axis=1))
    for i in range(n_phis):
        print(probs[i, :])

#%%
model = BayesianNetwork(nn_dims)

#%%
x_init = outcomes[1:10, 1:10, :]
print(model.tabulate(jax.random.PRNGKey(0), x_init))


#%%
@jax.jit
def train_step(state, batch):
    x, labels = batch
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, x)
        loss = optax.softmax_cross_entropy(
            logits,
            # optax.smooth_labels(labels, 0.01)
            labels
        ).mean(axis=(0, 1))
        return loss
    loss_val_grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = loss_val_grad_fn(state.params)

    state = state.apply_gradients(grads=grads)
    return state, loss


#%%
def create_train_state(model, init_key, x, learning_rate):
    params = model.init(init_key, x)['params']
    print("initial parameters", params)
    tx = optax.adam(learning_rate=learning_rate)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state


init_key = jax.random.PRNGKey(time.time_ns())
state = create_train_state(model, init_key, x_init, learning_rate=lr)
# del init_key

#%%
metrics = []
for i in (pbar := tqdm.tqdm(range(n_steps), disable=(not progress), mininterval=0.333)):
    # generate batch
    key = jax.random.PRNGKey(time.time_ns())
    subkeys = jax.random.split(key, num=2)

    inds = jax.random.randint(subkeys[0], minval=0, maxval=n_phis, shape=(batch_phis,))

    x = outcomes[inds, :, :]
    idx = jax.random.randint(subkeys[1], minval=0, maxval=n_shots, shape=(batch_phis, batch_shots, 1))
    x = jnp.take_along_axis(x, idx, axis=1)

    y = jnp.repeat(jnp.expand_dims(labels[inds], 1), repeats=batch_shots, axis=1)
    batch = (x, y)

    state, loss = train_step(state, batch)
    if progress:
        pbar.set_description(f"Step {i} | FI: {loss:.10f}", refresh=False)

    metrics.append(dict(step=i, loss=loss))

metrics = pd.DataFrame(metrics)

#%%
fig, ax = plt.subplots()
ax.plot(metrics.step, metrics.loss)

if show:
    plt.show()
if save:
    io.save_figure(fig, filename="loss.png")

#%%
state_params = state.params
# model.apply(state_params, x_init)

if input_type == "value":
    bit_strings = jnp.expand_dims(jnp.arange(n ** 2), 1)
elif input_type == "bits":
    bit_strings = jnp.expand_dims(jnp.arange(n ** 2), 1).astype(jnp.uint8)
    bit_strings = jnp.unpackbits(bit_strings, axis=1, bitorder='big')[:, -n:]
else:
    raise ValueError

pred = state.apply_fn({'params': state.params}, bit_strings)
pred = nn.activation.softmax(jnp.exp(pred), axis=-1)

colors = sns.color_palette('deep', n_colors=8)

fig, axs = plt.subplots(nrows=2, sharex=True)
ax = axs[0]
for i in range(bit_strings.shape[0]):
    print(i)
    ax.plot(jnp.linspace(phi_range[0], phi_range[1], pred.shape[1]), pred[i, :], ls='-', color=colors[i], label=f'Pr({i})')
ax.legend()

ax = axs[1]
for i in range(bit_strings.shape[0]):
    ax.plot(jnp.linspace(phi_range[0], phi_range[1], n_phis), probs[:, i], ls='--', color=colors[i], label=f'Pr({i}) Truth')
ax.set(xlabel='Phi Prediction', ylabel="Probability")

if show:
    plt.show()
if save:
    io.save_figure(fig, filename="probs.png")

#%% get relative frequencies
fig, axs = plt.subplots()

val = jnp.packbits(full_samples, axis=2, bitorder='little').squeeze()
# val = jnp.packbits(outcomes, axis=2, bitorder='little').squeeze()
rel_freq = jnp.stack([jnp.count_nonzero(val == m, axis=1) for m in range(n**2)], axis=1)

sns.heatmap(rel_freq)
plt.show()

#%%
likelihood = rel_freq / n_shots

pred = state.apply_fn({'params': state.params}, bit_strings)
pred = nn.activation.softmax(pred, axis=-1)
posterior = jnp.swapaxes(pred, 0, 1)

_outer = jnp.einsum('jm,km->jkm', likelihood, posterior)
A_jk = jnp.eye(n_grid, n_phis) - _outer.sum(axis=2)

# compute and sort eigenvectors by eigenvalue
w, v = jnp.linalg.eig(A_jk)
idx = np.argsort(w)
w = w[idx]
v = v[:, idx]

prior = v[:, 0].real
assert jnp.isclose(w[0], 0), "eigen-relation for prior not satisfied"
print(-prior)

#
fig, ax = plt.subplots()
sns.heatmap(A_jk)
plt.show()

#%%
fig, axs = plt.subplots(nrows=3)
colors = sns.color_palette('crest', as_cmap=True)
z = jnp.linspace(*phi_range, n_grid)

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

        ax.plot(z, pred, color=colors(k / n_phis))
        ax.axvline(phi, color=colors(k/n_phis), ls='--', alpha=0.4)

if show:
    plt.show()
if save:
    io.save_figure(fig, filename="sequence.png")
