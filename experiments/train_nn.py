import tqdm
import matplotlib.pyplot as plt
from typing import Sequence

import jax
import jax.numpy as jnp
import optax

from queso.estimators.ff import RegressionEstimator
from queso.io import IO
import h5py

from queso.utils import shots_to_counts


# %%
def train_nn(
    io: IO,
    key: jax.random.PRNGKey,
    nn_dims: Sequence[int],
    batch_size: int = 10,
    n_epochs: int = 100,
    lr: float = 1e-2,
    n_batches: int = 10,
    plot: bool = False,
    progress: bool = True,
):
    # %% hyperparameters

    # %% extract data from H5 file
    hf = h5py.File(io.path.joinpath("circ.h5"), "r")
    print(hf.keys())

    shots = jnp.array(hf.get("shots"))
    phis = jnp.array(hf.get("phis"))

    hf.close()

    # %%
    counts = shots_to_counts(shots, phis)

    # %%
    model = RegressionEstimator(nn_dims)

    x = counts
    y = phis

    # %% create batches
    batch_inds = jax.random.shuffle(key, jnp.array(list(range(x.shape[0]))))[
        :batch_size
    ]

    x_batch = x[batch_inds]
    y_batch = y[batch_inds]

    params = model.init(key, x)
    pred_batch = model.apply(params, x_batch)
    print(pred_batch)

    # %% define mean-squared error as the loss function
    def mse(params, x, y):
        def squared_error(x, y):
            pred = model.apply(params, x)
            return jnp.inner(y - pred, y - pred) / 2.0

        return jnp.mean(jax.vmap(squared_error)(x, y), axis=0)

    # %% audition the loss function
    print(mse(params, x_batch, y_batch))

    # %% audition value and grad calculations
    loss_val_grad = jax.value_and_grad(mse)
    print(loss_val_grad(params, x, y))

    # %% # JIT training step
    @jax.jit
    def step_nn(params, x, y, opt_state):
        val, grads = loss_val_grad(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return val, params, updates, opt_state

    # %% initialize optimizer
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    # %% run training loop
    losses = []
    for epoch in (pbar := tqdm.tqdm(range(n_epochs), disable=(not progress))):
        for batch in range(n_batches):
            key, subkey = jax.random.split(key)
            batch_inds = jax.random.randint(
                subkey, (batch_size,), minval=0, maxval=x.shape[0] - 1
            )

            x_batch = x[batch_inds]
            y_batch = y[batch_inds]
            val, params, updates, opt_state = step_nn(params, x, y, opt_state)
            # val, params, updates, opt_state = step_nn(params, x_batch, y_batch, opt_state)

        # val, params, updates, opt_state = step_nn(params, x, y, opt_state)
        losses.append(val)

        if progress:
            pbar.set_description(f" Epoch {epoch} | MSE: {val:.10f} | {x.shape}")

    losses = jnp.array(losses)
    nn_mse = losses
    pred = model.apply(params, x)

    if plot:
        # %% plot NN loss minimization
        fig, ax = plt.subplots()
        ax.plot(losses)
        ax.set(xlabel="Optimization step", ylabel="Loss")
        fig.show()
        io.save_figure(fig, filename="nn-loss.png")

        # %% run prediction on all phases
        fig, ax = plt.subplots()
        ax.scatter(y, pred)
        ax.set(xlabel=r"Ground truth, $\phi$", ylabel=r"Estimate, $\bar{\phi}$")
        fig.show()
        io.save_figure(fig, filename="ground-truth-phi-to-estimate.png")

    # %% save to H5 file
    metadata = dict(nn_dims=nn_dims, lr=lr, batch_size=batch_size, n_epochs=n_epochs)
    io.save_json(metadata, filename="nn-metadata.json")

    # io.save_json(serialization.to_state_dict(params), filename="nn-params.json")

    hf = h5py.File(io.path.joinpath("nn.h5"), "w")
    hf.create_dataset("nn_mse", data=nn_mse)
    hf.create_dataset("pred", data=pred)
    hf.close()
