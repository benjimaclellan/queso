import jax
import jax.numpy as jnp
import matplotlib.pyplot
import time

from queso.io import IO
from train_circuit import train_circuit
from train_nn import train_nn


n = 4
k = 4

io = IO(folder="test_hdf5")
io.path.mkdir(parents=True, exist_ok=True)

#%%
key = jax.random.PRNGKey(34)
train_circuit(io, n, k, key=key, n_phis=10, n_shots=100)
time.sleep(0.1)

#%%
key = jax.random.PRNGKey(41)
train_nn(io, key=key, batch_size=7, n_epochs=100, lr=1e-1)

#%%