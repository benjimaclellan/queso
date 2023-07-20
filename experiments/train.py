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
n_phis = 10
n_steps = 200
lr = 1e-3
key = jax.random.PRNGKey(time.time_ns())
progress = True
plot = True

#%%
if True:
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
n_shots = 50
n_shots_test = 10

#%%
if True:
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
key = jax.random.PRNGKey(time.time_ns())
n_steps = 5000
n_grid = n_phis  # todo: make more general - not requiring matching training phis and grid
nn_dims = [32, 32, n_grid]
lr = 1e-2
batch_phis = 128
batch_shots = 36
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
        batch_phis=batch_phis,
        batch_shots=batch_shots,
        plot=plot,
        progress=progress,
        from_checkpoint=from_checkpoint,
    )

# %%

