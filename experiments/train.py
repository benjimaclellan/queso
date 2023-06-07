import jax
import jax.numpy as jnp
import matplotlib.pyplot
import time

from queso.io import IO
from train_circuit import train_circuit
from train_nn import train_nn


n = 6
k = 6

io = IO(folder="test_hdf5")
io.path.mkdir(parents=True, exist_ok=True)

#%%
# key = jax.random.PRNGKey(34)
# train_circuit(io, n, k, key=key, n_phis=50, n_shots=1000)
# time.sleep(0.1)

#%%
key = jax.random.PRNGKey(time.time_ns())
train_nn(io, key=key, nn_dims=(2**n, 10, 8, 1), batch_size=45, n_batches=10, n_epochs=200, lr=1e-2)

#%%
