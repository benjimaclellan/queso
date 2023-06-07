import itertools
import tqdm
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax
from flax import serialization

from queso.nn import RegressionEstimator
from queso.io import IO
import h5py


def shots_to_counts(shots, phis):
    bin_str = [''.join(p) for p in itertools.product('01', repeat=shots.shape[2])]
    counts = []
    for i, phi in enumerate(phis):
        basis, count = jnp.unique(shots[i, :, :], return_counts=True, axis=0)
        c = {''.join([str(j) for j in basis[i]]): count[i].item() for i in range(len(count))}
        cts = [c.get(b, 0) for b in bin_str]
        counts.append(cts)
    counts = jnp.array(counts)
    return counts


#%%
def train_nn(
    io: IO,
    key: jax.random.PRNGKey,
    batch_size: int = 10,
    n_epochs: int = 100,
    lr: float = 1e-2,
    n_batches: int = 10
):

    #%% hyperparameters
    progress = True

    # %% extract data from H5 file
    hf = h5py.File(io.path.joinpath("circuit.h5"), 'r')
    print(hf.keys())

    shots = jnp.array(hf.get('shots'))
    phis = jnp.array(hf.get('phis'))

    hf.close()

    #%%
    counts = shots_to_counts(shots, phis)

    #%%
    n = shots.shape[1]  # number of qubits
    nn_dims = (2 ** n, 10, 10, 1)
    model = RegressionEstimator(nn_dims)

    x = counts
    y = phis

    #%% create batches

    batch_inds = jax.random.shuffle(key, jnp.array(list(range(x.shape[0]))))[:batch_size]

    x_batch = x[batch_inds]
    y_batch = y[batch_inds]

    params = model.init(key, x)
    pred_batch = model.apply(params, x_batch)
    print(pred_batch)

    #%% define mean-squared error as the loss function
    def mse(params, x, y):
        def squared_error(x, y):
            pred = model.apply(params, x)
            return jnp.inner(y-pred, y-pred) / 2.0
        return jnp.mean(jax.vmap(squared_error)(x, y), axis=0)

    #%% audition the loss function
    print(mse(params, x_batch, y_batch))

    #%% audition value and grad calculations
    loss_val_grad = jax.value_and_grad(mse)
    print(loss_val_grad(params, x, y))

    #%% # JIT training step
    @jax.jit
    def step_nn(params, x, y, opt_state):
        val, grads = loss_val_grad(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return val, params, updates, opt_state

    #%% initialize optimizer
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    #%% run training loop
    losses = []
    for epoch in range(n_epochs):
        for batch in range(n_batches):
            # batch_inds = jax.random.shuffle(key, jnp.array(list(range(x.shape[0]))))[:batch_size]
            batch_inds = jax.random.randint(key, (batch_size,), minval=0, maxval=x.shape[0]-1)

            x_batch = x[batch_inds]
            y_batch = y[batch_inds]
            val, params, updates, opt_state = step_nn(params, x_batch, y_batch, opt_state)

        # val, params, updates, opt_state = step_nn(params, x, y, opt_state)
        print(f"Epoch {epoch} | MSE: {val} | {x.shape}")
        losses.append(val)

    losses = jnp.array(losses)
    nn_mse = losses

    #%% plot NN loss minimization
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set(xlabel="Optimization step", ylabel="Loss")
    fig.show()
    io.save_figure(fig, filename="nn-loss.png")

    #%% run prediction on all phases
    pred = model.apply(params, x)
    fig, ax = plt.subplots()
    ax.set(xlabel=r"Ground truth, $\phi$", ylabel=r"Estimate, $\bar{\phi}$")
    ax.scatter(y, pred)
    fig.show()
    io.save_figure(fig, filename="ground-truth-phi-to-estimate.png")

    # %% save to H5 file
    metadata = dict(n=n, nn_dims=nn_dims, lr=lr, batch_size=batch_size, n_epochs=n_epochs)
    io.save_json(metadata, filename="nn-metadata.json")

    io.save_json(serialization.to_state_dict(params), filename="nn-params.json")

    hf = h5py.File(io.path.joinpath("nn.h5"), 'w')
    hf.create_dataset('nn_mse', data=nn_mse)
    hf.close()
