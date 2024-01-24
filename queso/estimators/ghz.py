import time
import os
import tqdm
import matplotlib.pyplot as plt
from itertools import cycle
import seaborn as sns
from typing import Sequence
import pandas as pd
import h5py
import argparse
import warnings

import jax
import jax.numpy as jnp

from queso.sensors.tc.utils import sample_int2bin
from queso.io import IO
from queso.configs import Configuration
from queso.utils import get_machine_info


# %%
def ghz_estimator(
    io: IO,
    config: Configuration,
    key: jax.random.PRNGKey,
    plot: bool = False,
    progress: bool = True,
):
    n_grid = config.n_grid
    lr = config.lr_nn
    l2_regularization = config.l2_regularization
    from_checkpoint = config.from_checkpoint
    logit_norm = False

    # %% extract data from H5 file
    t0 = time.time()

    hf = h5py.File(io.path.joinpath("train_samples.h5"), "r")
    shots = jnp.array(hf.get("shots"))
    counts = jnp.array(hf.get("counts"))
    probs = jnp.array(hf.get("probs"))
    phis = jnp.array(hf.get("phis"))
    hf.close()

    # %%
    n = shots.shape[2]
    n_shots = shots.shape[1]
    n_phis = shots.shape[0]

    # %%
    assert n_shots % batch_size == 0
    n_batches = n_shots // batch_size
    n_steps = n_epochs * n_batches

    # %%
    dphi = phis[1] - phis[0]
    phi_range = (jnp.min(phis), jnp.max(phis))

    grid = (phi_range[1] - phi_range[0]) * jnp.arange(n_grid) / (
            n_grid - 1
    ) + phi_range[0]
    index = jnp.stack([jnp.argmin(jnp.abs(grid - phi)) for phi in phis])

    if n_phis != n_grid:
        warnings.warn("Grid and training data do not match. untested behaviour.")

    labels = jax.nn.one_hot(index, num_classes=n_grid)

    model = BayesianDNNEstimator(nn_dims)

    x = shots
    y = labels