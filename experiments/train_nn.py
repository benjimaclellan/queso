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
from flax.training.train_state import TrainState

from queso.estimators.flax.dnn import BayesianDNNEstimator
from queso.io import IO
from queso.utils import get_machine_info


# %%
def train_nn(
    io: IO,
    key: jax.random.PRNGKey,
    nn_dims: Sequence[int],
    n_steps: int = 50000,
    lr: float = 1e-2,
    plot: bool = False,
    progress: bool = True,
):

    # %% extract data from H5 file
    hf = h5py.File(io.path.joinpath("circ.h5"), "r")
    print(hf.keys())

    shots = jnp.array(hf.get("shots"))
    probs = jnp.array(hf.get("probs"))
    phis = jnp.array(hf.get("phis"))

    hf.close()

    #%%
    n = shots.shape[2]
    n_shots = shots.shape[1]
    n_phis = shots.shape[0]

    n_grid = 50

    batch_phis = 32
    batch_shots = 1

    #%%
    phi_range = (jnp.min(phis), jnp.max(phis))
    # delta_phi = (phi_range[1] - phi_range[0]) / (n_grid - 1)  # needed for proper normalization
    # index = jnp.floor(n_grid * (phis / (phi_range[1] - phi_range[0])))
    index = jnp.floor(n_grid * phis / (phi_range[1] - phi_range[0]))  #- 1 / 2
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
    @jax.jit
    def train_step(state, batch):
        x, labels = batch

        def loss_fn(params):
            logits = state.apply_fn({'params': params}, x)
            loss = optax.softmax_cross_entropy(
                logits,
                labels
            ).mean(axis=(0, 1))
            return loss

        loss_val_grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = loss_val_grad_fn(state.params)

        state = state.apply_gradients(grads=grads)
        return state, loss

    # %%
    def create_train_state(model, init_key, x, learning_rate):
        params = model.init(init_key, x)['params']
        print("initial parameters", params)
        tx = optax.adam(learning_rate=learning_rate)
        state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
        return state

    init_key = jax.random.PRNGKey(time.time_ns())
    state = create_train_state(model, init_key, x_init, learning_rate=lr)
    # del init_key

    # %%
    metrics = []
    for i in (pbar := tqdm.tqdm(range(n_steps), disable=(not progress), mininterval=0.333)):
        # generate batch
        key = jax.random.PRNGKey(time.time_ns())
        subkeys = jax.random.split(key, num=2)

        inds = jax.random.randint(subkeys[0], minval=0, maxval=n_phis, shape=(batch_phis,))

        x_batch = x[inds, :, :]
        idx = jax.random.randint(subkeys[1], minval=0, maxval=n_shots, shape=(batch_phis, batch_shots, 1))
        x_batch = jnp.take_along_axis(x_batch, idx, axis=1)

        y_batch = jnp.repeat(jnp.expand_dims(y[inds], 1), repeats=batch_shots, axis=1)
        batch = (x_batch, y_batch)

        state, loss = train_step(state, batch)
        if progress:
            pbar.set_description(f"Step {i} | FI: {loss:.10f}", refresh=False)

        metrics.append(dict(step=i, loss=loss))

    metrics = pd.DataFrame(metrics)

    #%%
    if plot:
        # %% plot probs and relative freqs
        _tmp = jnp.packbits(shots, axis=2, bitorder='little').squeeze()
        freqs = jnp.stack([jnp.count_nonzero(_tmp == m, axis=1) for m in range(n ** 2)], axis=1)

        fig, axs = plt.subplots(nrows=2)
        sns.heatmap(probs, ax=axs[0])
        sns.heatmap(freqs, ax=axs[1])

        # ax.set(xlabel="Measurement outcome", ylabel="Phase")
        fig.show()
        io.save_figure(fig, filename="probs.png")

        # %% plot NN loss minimization
        fig, ax = plt.subplots()
        ax.plot(metrics.step, metrics.loss)
        ax.set(xlabel="Optimization step", ylabel="Loss")
        fig.show()
        io.save_figure(fig, filename="nn-loss.png")

        # %% run prediction on all possible inputs
        bit_strings = jnp.expand_dims(jnp.arange(n ** 2), 1).astype(jnp.uint8)
        bit_strings = jnp.unpackbits(bit_strings, axis=1, bitorder='big')[:, -n:]

        pred = state.apply_fn({'params': state.params}, bit_strings)
        pred = nn.activation.softmax(jnp.exp(pred), axis=-1)

        fig, axs = plt.subplots(nrows=2, sharex=True)
        colors = sns.color_palette('deep', n_colors=bit_strings.shape[0])
        markers = cycle([".", "o", "v", "^", "<", ">"])
        ax = axs[0]
        for i in range(bit_strings.shape[0]):
            print(i)
            ax.plot(jnp.linspace(phi_range[0], phi_range[1], pred.shape[1]),
                    pred[i, :],
                    ls='',
                    marker=next(markers),
                    color=colors[i],
                    label=f'Pr({i})')
        ax.legend()
        io.save_figure(fig, filename="ground-truth-phi-to-estimate.png")

        plt.show()

    # %% save to H5 file
    metadata = dict(nn_dims=nn_dims, lr=lr,)
    io.save_json(metadata, filename="nn-metadata.json")

    # io.save_json(serialization.to_state_dict(params), filename="nn-params.json")

    hf = h5py.File(io.path.joinpath("nn.h5"), "w")
    hf.create_dataset("pred", data=state.params)
    hf.close()

    io.save_dataframe(metrics, filename="metrics.csv")


if __name__ == "__main__":
    #%%
    io = IO(folder="2023-07-05_nn-estimator-n2-k4")
    key = jax.random.PRNGKey(time.time_ns())

    n_steps = 50000
    lr = 1e-4
    plot = True
    progress = True

    n_grid = 50

    nn_dims = [16, 16, n_grid]
    #
    # #%%
    # train_nn(
    #     io=io,
    #     key=key,
    #     nn_dims=nn_dims,
    #     n_steps=n_steps,
    #     lr=lr,
    #     plot=plot,
    #     progress=progress,
    # )