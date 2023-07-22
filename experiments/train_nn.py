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
from queso.sensors.tc.sensor import sample_int2bin
from queso.io import IO
from queso.utils import get_machine_info

# %%
def train_nn(
    io: IO,
    key: jax.random.PRNGKey,  # todo: use provided key for reproducibility
    nn_dims: Sequence[int],
    n_steps: int = 50000,
    n_grid: int = 50,
    lr: float = 1e-2,
    n_epochs: int = 10,
    batch_size: int = 10,
    plot: bool = False,
    progress: bool = True,
    from_checkpoint: bool = True,
):

    # %% extract data from H5 file
    t0 = time.time()

    hf = h5py.File(io.path.joinpath("samples.h5"), "r")
    shots = jnp.array(hf.get("shots"))
    counts = jnp.array(hf.get("counts"))
    shots_test = jnp.array(hf.get("shots_test"))
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

    index = jnp.arange(n_grid)
    phis = (phi_range[1] - phi_range[0]) * index / (n_grid - 1) + phi_range[0]
    assert n_phis == n_grid

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
    def l2_loss(params, alpha):
        return alpha * (params ** 2).mean()

    @jax.jit
    def train_step(state, batch):
        x_batch, y_batch = batch

        def loss_fn(params):
            logits = state.apply_fn({'params': params}, x_batch)
            # loss = optax.softmax_cross_entropy(
            #     logits,
            #     y_batch
            # ).mean(axis=(0, 1))

            loss = -jnp.sum(y_batch[:, None, :] * jax.nn.log_softmax(logits, axis=-1), axis=-1).mean(axis=(0, 1))

            # loss += l2_loss()
            return loss

        loss_val_grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = loss_val_grad_fn(state.params)

        state = state.apply_gradients(grads=grads)
        return state, loss

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
        schedule = optax.polynomial_schedule(
            init_value=lr,
            end_value=lr**2,
            power=1,
            transition_steps=n_steps,
            # transition_begin=n_steps//2,
        )
        tx = optax.adam(learning_rate=schedule)
        # tx = optax.adamw(learning_rate=learning_rate, weight_decay=1e-5)

        state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
        return state

    init_key = jax.random.PRNGKey(time.time_ns())
    state = create_train_state(model, init_key, x_init, learning_rate=lr)
    # del init_key

    # %%
    keys = jax.random.split(key, (n_epochs))
    metrics = []
    pbar = tqdm.tqdm(total=n_epochs * n_batches, disable=(not progress), mininterval=0.333)
    for i in range(n_epochs):
        # shuffle shots
        subkeys = jax.random.split(keys[i], n_phis)
        x = jnp.stack([jax.random.permutation(subkey, x[k, :, :]) for k, subkey in enumerate(subkeys)])

        for j in range(n_batches):
            x_batch = x[:, j * batch_size : (j + 1) * batch_size, :]
            y_batch = y  # use all phases each batch, but not all shots per phase
            batch = (x_batch, y_batch)

            state, loss = train_step(state, batch)
            if progress:
                pbar.update()
                pbar.set_description(f"Epoch {i} | Batch {j} | Loss: {loss:.10f}", refresh=False)
            metrics.append(dict(step=i, loss=loss))

    pbar.close()
    metrics = pd.DataFrame(metrics)

    #%% compute posterior
    assert n_phis == n_grid

    # approx likelihood from relative frequencies
    freqs = counts / counts.sum(axis=1, keepdims=True)
    likelihood = freqs

    bit_strings = sample_int2bin(jnp.arange(2**n), n)
    pred = model.apply({'params': state.params}, bit_strings)
    pred = jax.nn.softmax(pred, axis=-1)
    posterior = pred

    lp = (likelihood @ posterior).T
    a_jk = jnp.eye(n_phis, n_grid) - lp

    eigenvalues, eigenvectors = jnp.linalg.eig(a_jk)
    prior = jnp.abs(eigenvectors[:, 0])
    print(eigenvalues[0])

    assert jnp.all(eigenvalues[0] <= eigenvalues)  # ensure eigenvalue sorting is correct

    # idx = eigenvalues.real.argsort(order="")
    # eigenvalues = eigenvalues[idx]
    # eigenvectors = eigenvectors[:, idx]

    # eigenvalues[-1].real
    # prior = eigenvectors[:, -1].real

    #%%
    if plot:
        #%% plot prior
        fig, ax = plt.subplots()
        ax.stem(prior)
        fig.show()
        io.save_figure(fig, filename="prior.png")

        # %% plot NN loss minimization
        fig, ax = plt.subplots()
        ax.plot(metrics.step, metrics.loss)
        ax.set(xlabel="Optimization step", ylabel="Loss")
        fig.show()
        io.save_figure(fig, filename="nn-loss.png")

        # %% run prediction on all possible inputs
        bit_strings = jnp.expand_dims(jnp.arange(2 ** n), 1).astype(jnp.uint8)
        bit_strings = jnp.unpackbits(bit_strings, axis=1, bitorder='big')[:, -n:]

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

            xdata = jnp.linspace(phi_range[0], phi_range[1], probs.shape[0], endpoint=False)
            axs[1].plot(xdata, probs[:, i], color=colors[i])
            axs[2].plot(xdata, freqs[:, i], color=colors[i], ls='--', alpha=0.3)

        axs[-1].set(xlabel=r"$\phi_j$")
        axs[0].set(ylabel=r"Posterior distribution, Pr($\phi_j | b_i$)")
        io.save_figure(fig, filename="posterior-dist.png")

        plt.show()

    # %% save to disk
    metadata = dict(nn_dims=nn_dims, lr=lr, time=time.time() - t0)
    io.save_json(metadata, filename="nn-metadata.json")
    io.save_csv(metrics, filename="metrics")

    #%%
    info = get_machine_info()
    io.save_json(info, filename="machine-info.json")

    #%%
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

    #%%
    ckpt = {'params': state.params, 'nn_dims': nn_dims}
    ckpt_dir = io.path.joinpath("ckpts")

    ckptr = Checkpointer(PyTreeCheckpointHandler())  # A stateless object, can be created on the fly.
    ckptr.save(ckpt_dir, ckpt, save_args=orbax_utils.save_args_from_target(ckpt), force=True)
    restored = ckptr.restore(ckpt_dir, item=None)

    #%%


if __name__ == "__main__":
    #%%
    # io = IO(folder="2023-07-06_nn-estimator-n1-k1")
    # io = IO(folder="2023-07-11_calibration-samples-n2-ghz-backup")
    # io = IO(folder="2023-07-13_calibration-samples-n1-ghz")
    io = IO(folder=f"2023-07-18_fig-example-n{2}-k{1}", include_date=False)

    key = jax.random.PRNGKey(time.time_ns())

    n_steps = 3000
    lr = 1e-2
    batch_phis = 128
    batch_shots = 36
    plot = True
    progress = True
    from_checkpoint = False

    n_grid = 100

    nn_dims = [16, 16, n_grid]

    #%%
    train_nn(
        io=io,
        key=key,
        nn_dims=nn_dims,
        n_steps=n_steps,
        n_grid=n_grid,
        lr=lr,
        batch_phis=batch_phis,
        batch_shots=batch_shots,
        plot=plot,
        progress=progress,
        from_checkpoint=from_checkpoint,
    )
