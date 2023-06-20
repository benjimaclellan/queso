import jax
import jax.numpy as jnp
import matplotlib.pyplot
import time
import h5py
import matplotlib.pyplot as plt

from queso.io import IO
from train_circuit import train_circuit
from train_nn import train_nn


n = 4
k = 4

io = IO(folder=f"nn-estimator-n{n}-k{k}", include_date=True)
io.path.mkdir(parents=True, exist_ok=True)

# %%
key = jax.random.PRNGKey(time.time_ns())
train_circuit(io, n, k, key=key, n_steps=1000, lr=1e-1, n_phis=500, n_shots=5000)
time.sleep(0.1)

# #%%
# key = jax.random.PRNGKey(time.time_ns())
# train_nn(
#     io, key=key, nn_dims=(2**n, 20, 20, 20, 1),
#     batch_size=100, n_batches=1, n_epochs=10000, lr=1e-3
# )

# %%
