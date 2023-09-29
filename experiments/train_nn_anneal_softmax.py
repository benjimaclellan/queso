import copy
import time
import tqdm
import matplotlib.pyplot as plt
from itertools import cycle
import seaborn as sns
from typing import Sequence
import pandas as pd
import h5py
import argparse
import warnings

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state, orbax_utils
from orbax.checkpoint import PyTreeCheckpointer, Checkpointer, CheckpointManager, CheckpointManagerOptions, PyTreeCheckpointHandler

from queso.estimators.flax.dnn import BayesianDNNEstimator
from queso.sensors.tc.sensor import sample_int2bin
from queso.io import IO
from queso.configs import Configuration
from queso.utils import get_machine_info


# %%
io = IO(folder="test_multi_shot_training", include_date=True)
config = Configuration()
key = jax.random.PRNGKey(1234)
plot = False
progress = True

#%%
n_grid = config.n_grid
n_grid = 200
nn_dims = config.nn_dims + [n_grid]
lr = config.lr_nn
l2_regularization = config.l2_regularization
n_epochs = config.n_epochs
batch_size = config.batch_size
from_checkpoint = config.from_checkpoint
logit_norm = False

# %% extract data from H5 file
t0 = time.time()

hf = h5py.File(io.path.joinpath("train_samples.h5"), "r")
shots = jnp.array(hf.get("shots"))
counts = jnp.array(hf.get("counts"))
probs = jnp.array(hf.get("probs"))
phis = jnp.array(hf.get("phis"))
hf.close()

#%%
n = shots.shape[2]
n_shots = shots.shape[1]
n_phis = shots.shape[0]

#%%
assert n_shots % batch_size == 0
n_batches = n_shots // batch_size
n_steps = n_epochs * n_batches

#%%
dphi = phis[1] - phis[0]
phi_range = (jnp.min(phis), jnp.max(phis))

grid = (phi_range[1] - phi_range[0]) * jnp.arange(n_grid) / (n_grid - 1) + phi_range[0]
index = jnp.stack([jnp.argmin(jnp.abs(grid - phi)) for phi in phis])

if n_phis != n_grid:
    warnings.warn("Grid and training data do not match. untested behaviour.")

labels = jax.nn.one_hot(index, num_classes=n_grid)

print(index)
print(labels.sum(axis=0))

# %%
model = BayesianDNNEstimator(nn_dims)

x = shots
y = labels

#%%
x_init = x[1:10, 1:10, :]
print(model.tabulate(jax.random.PRNGKey(0), x_init))


# %%
def l2_loss(w, alpha):
    return alpha * (w ** 2).mean()


@jax.jit
def train_step(state, batch):
    x_batch, y_batch = batch

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, x_batch)

        log_probs = jax.nn.log_softmax(logits_multi, axis=-1)

        # standard cross-entropy
        loss = -jnp.sum(y_batch[:, None, :] * log_probs, axis=-1).mean(axis=(0, 1))

        loss += sum(
            l2_loss(w, alpha=l2_regularization)
            for w in jax.tree_leaves(params)
        )
        return loss

    loss_val_grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = loss_val_grad_fn(state.params)

    state = state.apply_gradients(grads=grads)
    return state, loss


#%%
def shuffle_shots(shots, key=None):
    if key is None:
        key = jax.random.PRNGKey(time.time_ns())
    shots_multi = copy.deepcopy(shots)
    shots_multi = jax.random.shuffle(key, shots_multi, axis=1)
    splits = jnp.split(
        shots_multi[:, :jnp.floor(shots_multi.shape[1]/n_splits).astype('int') * n_splits, :],
        n_splits,
        axis=1
    )
    shots_multi = jnp.stack(splits, axis=0)
    return shots_multi


# %%
def create_train_state(model, init_key, x, learning_rate):
    if from_checkpoint:
        ckpt_dir = io.path.joinpath("ckpts")
        ckptr = Checkpointer(PyTreeCheckpointHandler())  # A stateless object, can be created on the fly.
        restored = ckptr.restore(ckpt_dir, item=None)
        params = restored['params']
        print(f"Loading parameters from checkpoint: {ckpt_dir}")
    else:
        params = model.init(init_key, x)['params']
        print(f"Random initialization of parameters")

    # print("Initial parameters", params)
    # schedule = optax.constant_schedule(lr)
    schedule = optax.polynomial_schedule(
        init_value=lr,
        end_value=lr**2,
        power=1,
        transition_steps=n_steps//4,
        transition_begin=3 * n_steps//2,
    )
    tx = optax.adam(learning_rate=schedule)
    # tx = optax.adamw(learning_rate=learning_rate, weight_decay=1e-5)

    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state

init_key = jax.random.PRNGKey(time.time_ns())
state = create_train_state(model, init_key, x_init, learning_rate=lr)
# del init_key

#%%
lr = 0.1
n_epochs = 100
n_splits = 2

keys = jax.random.split(key, (n_epochs))
metrics = []
pbar = tqdm.tqdm(total=n_epochs, disable=(not progress), mininterval=0.333)
for i in range(n_epochs):  # number of epochs
    shots_multi = shuffle_split_stack_shots(shots, n_splits=n_splits)

    batch = (shots_multi, y)

    state, loss = train_step(state, batch)
    if progress:
        pbar.update()
        pbar.set_description(f"Epoch {i} | Loss: {loss:.10f}", refresh=False)
    metrics.append(dict(step=i, loss=loss))

pbar.close()
metrics = pd.DataFrame(metrics)

#%%
fig, ax = plt.subplots()
ax.plot(metrics.step, metrics.loss)
io.save_figure(fig, filename='nn_loss.png')

#%%
logits_multi = state.apply_fn({'params': state.params}, shots_multi)
# log_probs_multi = jax.nn.log_softmax(logits_multi, axis=-1)
# probs = jax.nn.softmax(logits_multi, axis=-1).squeeze()
prod_logits_multi = prod(logits_multi)

fig, ax = plt.subplots()
ax.plot(prod_logits_multi[0, 0, :])
io.save_figure(fig, filename="test_probs.png")

#%%
x_batch = x[:, 0:batch_size, :]
y_batch = y
batch = (x_batch, y_batch)

state, loss = train_step(state, batch)

#%%
hf = h5py.File(io.path.joinpath("nn.h5"), "w")
hf.create_dataset("grid", data=grid)
hf.close()

#%% compute posterior
# approx likelihood from relative frequencies
freqs = counts / counts.sum(axis=1, keepdims=True)
likelihood = freqs

bit_strings = sample_int2bin(jnp.arange(2**n), n)
pred = model.apply({'params': state.params}, bit_strings)
pred = jax.nn.softmax(pred, axis=-1)
posterior = pred


# %% save to disk
metadata = dict(nn_dims=nn_dims, lr=lr, time=time.time() - t0)
io.save_json(metadata, filename="nn-metadata.json")
io.save_csv(metrics, filename="metrics")

# %%
info = get_machine_info()
io.save_json(info, filename="machine-info.json")

# %%
ckpt = {'params': state.params, 'nn_dims': nn_dims}
ckpt_dir = io.path.joinpath("ckpts")

ckptr = Checkpointer(PyTreeCheckpointHandler())  # A stateless object, can be created on the fly.
ckptr.save(ckpt_dir, ckpt, save_args=orbax_utils.save_args_from_target(ckpt), force=True)
restored = ckptr.restore(ckpt_dir, item=None)

print(f"Finished training the estimator.")

#%%
if plot:
    #%% plot prior
    # fig, ax = plt.subplots()
    # ax.stem(prior)
    # fig.show()
    # io.save_figure(fig, filename="prior.png")

    # %% plot NN loss minimization
    fig, ax = plt.subplots()
    ax.plot(metrics.step, metrics.loss)
    ax.set(xlabel="Optimization step", ylabel="Loss")
    fig.show()
    io.save_figure(fig, filename="nn-loss.png")

    # %% run prediction on all possible inputs
    bit_strings = sample_int2bin(jnp.arange(2 ** n), n)
    pred = model.apply({'params': state.params}, bit_strings)
    pred = jax.nn.softmax(pred, axis=-1)

    fig, axs = plt.subplots(nrows=3, figsize=[9, 6], sharex=True)
    colors = sns.color_palette('deep', n_colors=bit_strings.shape[0])
    markers = cycle(["o", "D", 's', "v", "^", "<", ">", ])
    for i in range(bit_strings.shape[0]):
        ax = axs[0]
        xdata = jnp.linspace(phi_range[0], phi_range[1], pred.shape[1], endpoint=False)
        ax.plot(xdata,
                pred[i, :],
                ls='',
                marker=next(markers),
                color=colors[i],
                label=r"Pr($\phi_j | "+"b_{"+str(i)+"}$)")

        xdata = jnp.linspace(phi_range[0], phi_range[1], counts.shape[0], endpoint=False)
        # if not jnp.all(jnp.isnan(probs)).item():
        #     axs[1].plot(xdata, probs[:, i], color=colors[i])
        axs[2].plot(xdata, freqs[:, i], color=colors[i], ls='--', alpha=0.3)

    axs[-1].set(xlabel=r"$\phi_j$")
    axs[0].set(ylabel=r"Posterior distribution, Pr($\phi_j | b_i$)")
    io.save_figure(fig, filename="posterior-dist.png")

    plt.show()



