import jax
import jax.numpy as jnp

from queso.io import IO
from train_circuit import train_circuit
from train_nn import train_nn


n = 6
k = 6

io = IO(folder="test_hdf5")
io.path.mkdir(parents=True, exist_ok=True)

key = jax.random.PRNGKey(1234)
train_circuit(io, n, k, key=key, n_phis=10, n_shots=100)

key = jax.random.PRNGKey(4321)
train_nn(io, key=key, batch_size=4, n_epochs=10)
