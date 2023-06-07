import itertools
import time
import tqdm
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax

from queso.tc.sensor import Sensor
from queso.estimator import RegressionEstimator
from queso.io import IO
from train_circuit import train_circuit


n = 6
k = 6

io = IO(folder="test_hdf5")
io.path.mkdir(parents=True, exist_ok=True)

key = jax.random.PRNGKey(1234)
train_circuit(io, n, k, key)

