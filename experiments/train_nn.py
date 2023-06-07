import itertools
import time
import tqdm
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax

from queso.estimator import RegressionEstimator
from queso.io import IO
import h5py


io = IO(folder="test_hdf5")
io.path.mkdir(parents=True, exist_ok=True)

# %%
hf = h5py.File(io.path.joinpath("test.h5"), 'r')
print(hf.keys())