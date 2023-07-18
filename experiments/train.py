import jax
import jax.numpy as jnp
import matplotlib.pyplot
import time
import h5py
import matplotlib.pyplot as plt

from queso.io import IO
from train_circuit import train_circuit
from train_nn import train_nn

#%%
n = 2
k = 1

io = IO(folder=f"fig-example-n{n}-k{k}", include_date=True)
io.path.mkdir(parents=True, exist_ok=True)

# %%
key = jax.random.PRNGKey(time.time_ns())
train_circuit(
    io=io,
    n=n,
    k=k,
    key=key,
    phi_range=(-jnp.pi/4, jnp.pi/4),
    n_steps=1000,
    lr=1e-3,
    n_phis=100,
    n_shots=1000,
    progress=True,
    plot=True,
)

#%%
key = jax.random.PRNGKey(time.time_ns())
nn_dims = [16, 16, 100]
n_steps = 3000
n_grid = 100
lr = 1e-2
batch_phis = 128
batch_shots = 36
plot = True
progress = True
from_checkpoint = False
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
