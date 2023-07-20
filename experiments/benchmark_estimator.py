#%%
import time
import tqdm
import matplotlib.pyplot as plt
from itertools import cycle
import seaborn as sns
from typing import Sequence
import pandas as pd
import h5py

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state, orbax_utils
from orbax.checkpoint import PyTreeCheckpointer, Checkpointer, \
    CheckpointManager, CheckpointManagerOptions, PyTreeCheckpointHandler

from queso.estimators.flax.dnn import BayesianDNNEstimator
from queso.io import IO
from queso.utils import get_machine_info


#%%
io = IO(folder="2023-07-18_fig-example-n2-k1", verbose=False)
# io = IO(folder="2023-07-18_cr-ansatz-n6-k6", verbose=False)

#%%
hf = h5py.File(io.path.joinpath("circ.h5"), "r")
print(hf.keys())
shots = jnp.array(hf.get("shots"))
probs = jnp.array(hf.get("probs"))
phis = jnp.array(hf.get("phis"))
hf.close()

n = shots.shape[2]
n_phis = shots.shape[0]
phi_range = (jnp.min(phis), jnp.max(phis))

#%%
ckpt_dir = io.path.joinpath("ckpts")
ckptr = Checkpointer(PyTreeCheckpointHandler())  # A stateless object, can be created on the fly.
restored = ckptr.restore(ckpt_dir, item=None)
nn_dims = restored['nn_dims']

#%%
key = jax.random.PRNGKey(0)
model = BayesianDNNEstimator(nn_dims)

#%%
n_trials = 10

phis_inds = jnp.array([0, 50])
phis_true = phis[phis_inds]

# n_sequences = (1, 10, 100, 1000)
n_sequences = jnp.round(jnp.logspace(0, 2, 20)).astype("int")
# n_sequences = jnp.round(jnp.linspace(1, 10**3, 20)).astype("int")
n_sequence_max = n_sequences[-1]


#%%
def select_sample_sequence(shots, key):
    shot_inds = jax.random.randint(key, shape=(n_sequence_max,), minval=0, maxval=shots.shape[1])
    shots_phis = shots[phis_inds, :, :]
    sequences = shots_phis[:, shot_inds, :]
    return sequences


key = jax.random.PRNGKey(0)
# key = jax.random.PRNGKey(time.time_ns())
keys = jax.random.split(key, n_trials)

sequences = jnp.stack([select_sample_sequence(shots, key) for key in keys], axis=0)
assert sequences.shape == (n_trials, len(phis_true), n_sequence_max, n)

#%%
pred = model.apply({'params': restored['params']}, sequences)
pred = nn.activation.softmax(pred, axis=-1)
print(pred.shape)
assert pred.shape == (n_trials, len(phis_true), n_sequence_max, n_phis)


#%%
def posterior_product(pred, n_sequence):
    tmp = pred[:, :, :n_sequence, :]
    tmp = jnp.log(tmp).sum(axis=-2, keepdims=False)  # sum log posterior probs for each individual input sample
    tmp = jnp.exp(tmp - tmp.max(axis=-1, keepdims=True))  # help with underflow in normalization
    posteriors = tmp / tmp.sum(axis=-1, keepdims=True)
    return posteriors


def estimate(posteriors, phis):
    return phis[jnp.argmax(posteriors, axis=-1)]


@jax.jit
def bias(phi_estimates, phis_true):
    biases = phi_estimates - phis_true[None, :, None]
    return biases


@jax.jit
def variance(posteriors, phi_estimates, phis):
    variances = (posteriors * jnp.power(phi_estimates[:, :, :, None] - phis[None, None, None, :], 2)).sum(axis=-1)
    return variances


posteriors = jnp.stack([posterior_product(pred, n_sequence) for n_sequence in n_sequences], axis=2)
assert posteriors.shape == (n_trials, len(phis_true), len(n_sequences), n_phis)

phi_estimates = estimate(posteriors, phis)
biases = bias(phi_estimates, phis_true)
variances = variance(posteriors, phi_estimates, phis)

#%% plot posterior, bias, and variance for one phase
fig, axs = plt.subplots(nrows=3, figsize=(6.5, 6.0))
colors = sns.color_palette('crest', n_colors=n_sequences.shape[0])
markers = ["o", "D", 's', "v", "^", "<", ">", ]

k = 0  # which phi to plot

for i, n_sequence in enumerate(n_sequences):
    ax = axs[0]
    ax.axvline(phis_true[k], color=colors[0], ls='--', alpha=0.7)
    p = posteriors[0, k, i, :]
    ax.plot(
        jnp.linspace(phi_range[0], phi_range[1], n_phis),
        p / jnp.max(p),
        ls='',
        marker=markers[i % len(markers)],
        color=colors[i],
        # alpha=(0.1 + i / len(n_sequences) * 0.8),
        alpha=(i+1) / len(n_sequences),
        markersize=3,
        label=r"$\phi_{true}=$" + f"{phis_true[k] / jnp.pi:0.2f}$\pi$",
    )

ax = axs[1]
ax.plot(
    n_sequences,
    biases[:, k, :].mean(axis=0),
    color=colors[0],
    ls=':',
    marker=markers[0],
)
ax.set(xscale="log")

ax = axs[2]
ax.plot(
    n_sequences,
    variances[:, k, :].mean(axis=0),
    color=colors[0],
    ls=':',
    marker=markers[0],
)
ax.set(xscale="log")

axs[0].set(xlabel="$\phi_j$", ylabel=r"p($\phi_j | \vec{s}$)")
axs[1].set(xlabel="Sequence length, $m$", ylabel=r"Bias, $\langle \hat{\varphi} - \varphi \rangle$")

fig.tight_layout()
plt.show()

