import jax
import jax.numpy as jnp
import matplotlib.pyplot
import time
import h5py
import matplotlib.pyplot as plt

from queso.io import IO
from train_circuit import train_circuit
from sample_circuit import sample_circuit
from train_nn import train_nn

#%%
n = 4
k = 4

io = IO(folder=f"estimator-performance-n{n}-k{k}", include_date=True)
io.path.mkdir(parents=True, exist_ok=True)

# %%
phi_range = (0, jnp.pi / 2)
n_phis = 100
n_steps = 20000
lr = 1e-3
key = jax.random.PRNGKey(time.time_ns())
progress = True
plot = True

#%%
if False:
    train_circuit(
        io=io,
        n=n,
        k=k,
        key=key,
        phi_range=phi_range,
        n_phis=n_phis,
        n_steps=n_steps,
        lr=lr,
        contractor="plain",
        progress=progress,
        plot=plot,
    )

#%%
n_shots = 5000
n_shots_test = 1000

#%%
if False:
    sample_circuit(
        io=io,
        n=n,
        k=k,
        key=key,
        phi_range=phi_range,
        n_phis=n_phis,
        n_shots=n_shots,
        n_shots_test=n_shots_test,
        plot=plot,
    )

#%%
# key = jax.random.PRNGKey(time.time_ns())
key = jax.random.PRNGKey(0)

# n_steps = 50000
n_epochs = 100
batch_size = 50
n_grid = n_phis  # todo: make more general - not requiring matching training phis and grid
nn_dims = [32, 32, 32, n_grid]
lr = 1e-3
plot = True
progress = True
from_checkpoint = False

#%%
if True:
    train_nn(
        io=io,
        key=key,
        nn_dims=nn_dims,
        n_steps=n_steps,
        n_grid=n_grid,
        lr=lr,
        n_epochs=n_epochs,
        batch_size=batch_size,
        # batch_phis=batch_phis,
        # batch_shots=batch_shots,
        plot=plot,
        progress=progress,
        from_checkpoint=from_checkpoint,
    )

# %%

