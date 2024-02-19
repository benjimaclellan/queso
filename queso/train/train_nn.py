# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright (c) 2022-2024 Benjamin MacLellan

import time
import os
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
from orbax.checkpoint import (
    PyTreeCheckpointer,
    Checkpointer,
    CheckpointManager,
    CheckpointManagerOptions,
    PyTreeCheckpointHandler,
)

from queso.estimators.flax.dnn import BayesianDNNEstimator
from queso.sensors.tc.utils import sample_int2bin
from queso.io import IO
from queso.configs import Configuration
from queso.utils import get_machine_info


# %%
def train_nn(
    io: IO,
    config: Configuration,
    key: jax.random.PRNGKey,
    plot: bool = False,
    progress: bool = True,
):
    """
    Trains a neural network based on the provided configuration.

    This function initializes a BayesianDNNEstimator, sets up the optimizer and loss function, and then performs the training steps.
    It also saves the training results and metadata, and optionally plots the training progress.

    Args:
        io (IO): An instance of the IO class for handling input/output operations.
        config (Configuration): An instance of the Configuration class containing the settings for the training.
        key (jax.random.PRNGKey): A random number generator key from JAX.
        plot (bool, optional): If True, plots of the training progress are generated and saved. Defaults to False.
        progress (bool, optional): If True, a progress bar is displayed during training. Defaults to True.

    Returns:
        None

    Raises:
        Warning: If the grid and training data do not match.
    """
    jax.config.update("jax_default_device", jax.devices(os.getenv("DEFAULT_DEVICE_TRAIN_NN", "cpu"))[0])

    # %%
    nn_dims = config.nn_dims + [config.n_grid]
    n_grid = config.n_grid
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

    # %%
    n = shots.shape[2]
    n_shots = shots.shape[1]
    n_phis = shots.shape[0]

    # %%
    assert n_shots % batch_size == 0
    n_batches = n_shots // batch_size
    n_steps = n_epochs * n_batches

    # %%
    dphi = phis[1] - phis[0]
    phi_range = (jnp.min(phis), jnp.max(phis))

    grid = (phi_range[1] - phi_range[0]) * jnp.arange(n_grid) / (
        n_grid - 1
    ) + phi_range[0]
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

    # mu, sig = 0.0, 0.05
    # g = (1/sig/jnp.sqrt(2 * jnp.pi)) * jnp.exp(- (jnp.linspace(-1, 1, n_grid) - mu) ** 2 / 2 / sig**2)
    # yg = jnp.fft.ifft(jnp.fft.fft(y, axis=1) * jnp.fft.fft(jnp.fft.fftshift(g)), axis=1).real
    # fig, ax = plt.subplots()
    # sns.heatmap(yg, ax=ax)
    # plt.show()

    # %%
    x_init = x[1:10, 1:10, :]
    print(model.tabulate(jax.random.PRNGKey(0), x_init))

    # %%
    def l2_loss(w, alpha):
        return alpha * (w**2).mean()

    @jax.jit
    def train_step(state, batch):
        x_batch, y_batch = batch

        def loss_fn(params):
            logits = state.apply_fn({"params": params}, x_batch)
            # loss = optax.softmax_cross_entropy(
            #     logits,
            #     y_batch
            # ).mean(axis=(0, 1))

            if logit_norm:
                eps = 1e-10
                tau = 10.0
                logits = (
                    (logits + eps)
                    / (jnp.sqrt((logits**2 + eps).sum(axis=-1, keepdims=True)))
                    / tau
                )

            # standard cross-entropy
            print("softmax", jax.nn.log_softmax(logits, axis=-1).shape)
            loss = -jnp.sum(
                y_batch[:, None, :] * jax.nn.log_softmax(logits, axis=-1), axis=-1
            ).mean(axis=(0, 1))

            # cross-entropy with ReLUmax instead of softmax
            # log_relumax = jnp.log(jax.nn.relu(logits) / jnp.sum(jax.nn.relu(logits), axis=-1, keepdims=True))
            # print('relumax', log_relumax)
            # loss = -jnp.sum(y_batch[:, None, :] * log_relumax, axis=-1).mean(axis=(0, 1))

            # MSE loss
            # loss = jnp.sum((y_batch[:, None, :] - jax.nn.softmax(logits, axis=-1))**2, axis=-1).mean(axis=(0, 1))

            # CE loss + convolution w/ Gaussian
            # loss = -jnp.sum(yg[:, None, :] * jax.nn.log_softmax(logits, axis=-1), axis=-1).mean(axis=(0, 1))

            loss += sum(
                l2_loss(w, alpha=l2_regularization) for w in jax.tree_leaves(params)
            )
            return loss

        loss_val_grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = loss_val_grad_fn(state.params)

        state = state.apply_gradients(grads=grads)
        return state, loss

    # %%
    def create_train_state(model, init_key, x, learning_rate):
        if from_checkpoint:
            ckpt_dir = io.path.joinpath("ckpts")
            ckptr = Checkpointer(
                PyTreeCheckpointHandler()
            )  # A stateless object, can be created on the fly.
            restored = ckptr.restore(ckpt_dir, item=None)
            params = restored["params"]
            print(f"Loading parameters from checkpoint: {ckpt_dir}")
        else:
            params = model.init(init_key, x)["params"]
            print(f"Random initialization of parameters")

        # print("Initial parameters", params)
        # schedule = optax.constant_schedule(lr)
        schedule = optax.polynomial_schedule(
            init_value=lr,
            end_value=lr**2,
            power=1,
            transition_steps=n_steps // 4,
            transition_begin=3 * n_steps // 2,
        )
        tx = optax.adam(learning_rate=schedule)
        # tx = optax.adamw(learning_rate=learning_rate, weight_decay=1e-5)

        state = train_state.TrainState.create(
            apply_fn=model.apply, params=params, tx=tx
        )
        return state

    init_key = jax.random.PRNGKey(time.time_ns())
    state = create_train_state(model, init_key, x_init, learning_rate=lr)
    # del init_key

    # %%
    x_batch = x[:, 0:batch_size, :]
    y_batch = y
    batch = (x_batch, y_batch)

    state, loss = train_step(state, batch)

    # %%
    keys = jax.random.split(key, (n_epochs))
    metrics = []
    pbar = tqdm.tqdm(
        total=n_epochs * n_batches, disable=(not progress), mininterval=0.333
    )
    for i in range(n_epochs):
        # shuffle shots
        # subkeys = jax.random.split(keys[i], n_phis)
        # x = jnp.stack([jax.random.permutation(subkey, x[k, :, :]) for k, subkey in enumerate(subkeys)])

        for j in range(n_batches):
            x_batch = x[:, j * batch_size : (j + 1) * batch_size, :]
            y_batch = y  # use all phases each batch, but not all shots per phase
            batch = (x_batch, y_batch)

            state, loss = train_step(state, batch)
            if progress:
                pbar.update()
                pbar.set_description(
                    f"Epoch {i} | Batch {j:04d} | Loss: {loss:.10f}", refresh=False
                )
            metrics.append(dict(step=i * n_batches + j, loss=loss))

    pbar.close()
    metrics = pd.DataFrame(metrics)

    # %%
    hf = h5py.File(io.path.joinpath("nn.h5"), "w")
    hf.create_dataset("grid", data=grid)
    hf.close()

    # %% compute posterior
    # approx likelihood from relative frequencies
    freqs = counts / counts.sum(axis=1, keepdims=True)
    likelihood = freqs

    bit_strings = sample_int2bin(jnp.arange(2**n), n)
    pred = model.apply({"params": state.params}, bit_strings)
    pred = jax.nn.softmax(pred, axis=-1)
    posterior = pred

    # lp = (likelihood @ posterior).T
    # a_jk = jnp.eye(n_phis, n_grid) - lp

    # eigenvalues, eigenvectors = jnp.linalg.eig(a_jk)
    # prior = jnp.abs(eigenvectors[:, 0])
    # print(eigenvalues[0])

    # assert jnp.all(eigenvalues[0] <= eigenvalues)  # ensure eigenvalue sorting is correct

    # idx = eigenvalues.real.argsort(order="")
    # eigenvalues = eigenvalues[idx]
    # eigenvectors = eigenvectors[:, idx]

    # eigenvalues[-1].real
    # prior = eigenvectors[:, -1].real

    # %% save to disk
    metadata = dict(nn_dims=nn_dims, lr=lr, time=time.time() - t0)
    io.save_json(metadata, filename="nn-metadata.json")
    io.save_csv(metrics, filename="metrics")

    # %%
    info = get_machine_info()
    io.save_json(info, filename="machine-info.json")

    # %%
    # ckpt = {'params': state, 'nn_dims': nn_dims}
    # ckpt_dir = io.path.joinpath("ckpts")
    #
    # orbax_checkpointer = PyTreeCheckpointer()
    # options = CheckpointManagerOptions(max_to_keep=2)
    # checkpoint_manager = CheckpointManager(ckpt_dir, orbax_checkpointer, options)
    # save_args = orbax_utils.save_args_from_target(ckpt)
    #
    # # doesn't overwrite
    # check = checkpoint_manager.save(0, ckpt, save_kwargs={'save_args': save_args})
    # print(check)
    # restored = checkpoint_manager.restore(0)

    # %%
    ckpt = {"params": state.params, "nn_dims": nn_dims}
    ckpt_dir = io.path.joinpath("ckpts")

    ckptr = Checkpointer(
        PyTreeCheckpointHandler()
    )  # A stateless object, can be created on the fly.
    ckptr.save(
        ckpt_dir, ckpt, save_args=orbax_utils.save_args_from_target(ckpt), force=True
    )
    restored = ckptr.restore(ckpt_dir, item=None)

    print(f"Finished training the estimator.")

    # %%
    if plot:
        # %% plot prior
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
        bit_strings = sample_int2bin(jnp.arange(2**n), n)
        pred = model.apply({"params": state.params}, bit_strings)
        pred = jax.nn.softmax(pred, axis=-1)

        fig, axs = plt.subplots(nrows=3, figsize=[9, 6], sharex=True)
        colors = sns.color_palette("deep", n_colors=bit_strings.shape[0])
        markers = cycle(
            [
                "o",
                "D",
                "s",
                "v",
                "^",
                "<",
                ">",
            ]
        )
        for i in range(bit_strings.shape[0]):
            ax = axs[0]
            xdata = jnp.linspace(
                phi_range[0], phi_range[1], pred.shape[1], endpoint=False
            )
            ax.plot(
                xdata,
                pred[i, :],
                ls="",
                marker=next(markers),
                color=colors[i],
                label=r"Pr($\phi_j | " + "b_{" + str(i) + "}$)",
            )

            xdata = jnp.linspace(
                phi_range[0], phi_range[1], counts.shape[0], endpoint=False
            )
            # if not jnp.all(jnp.isnan(probs)).item():
            #     axs[1].plot(xdata, probs[:, i], color=colors[i])
            axs[2].plot(xdata, freqs[:, i], color=colors[i], ls="--", alpha=0.3)

        axs[-1].set(xlabel=r"$\phi_j$")
        axs[0].set(ylabel=r"Posterior distribution, Pr($\phi_j | b_i$)")
        io.save_figure(fig, filename="posterior-dist.png")

        plt.show()


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="tmp")
    args = parser.parse_args()
    folder = args.folder

    io = IO(folder=f"{folder}")
    print(io)
    config = Configuration.from_yaml(io.path.joinpath("config.yaml"))
    key = jax.random.PRNGKey(config.seed)
    print(f"Training NN: {folder} | Devices {jax.devices()} | Full path {io.path}")
    print(f"Config: {config}")
    train_nn(io, config, key, progress=True, plot=True)
