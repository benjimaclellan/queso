# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright (c) 2022-2024 Benjamin MacLellan

import time
import os
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import argparse

import jax
import jax.numpy as jnp
import optax

from queso.sensors.tc.sensor import Sensor
from queso.sensors.tc.utils import sample_bin2int
from queso.io import IO
from queso.configs import Configuration


# %%
def sample_circuit_testing(
    io: IO,
    config: Configuration,
    key: jax.random.PRNGKey,
    plot: bool = False,
    progress: bool = True,
):
    """
    Samples a quantum circuit for testing based on the provided configuration and random key.

    This function initializes a sensor with the given configuration, samples the circuit,
    and saves the sampled data in an HDF5 file. The data is for testing estimators.

    Args:
        io (IO): An IO object for handling file operations.
        config (Configuration): A Configuration object containing the parameters for the circuit.
        key (jax.random.PRNGKey): A random key for JAX operations.
        plot (bool, optional): If True, plots the true probabilities and relative frequencies. Defaults to False.
        progress (bool, optional): If True, displays progress information. Defaults to True.

    Returns:
        None
    """

    n = config.n
    k = config.k
    phis_test = jnp.array(config.phis_test)
    n_shots_test = config.n_shots_test
    kwargs = dict(
        preparation=config.preparation,
        interaction=config.interaction,
        detection=config.detection,
        backend=config.backend,
        n_ancilla=config.n_ancilla,
        gamma_dephasing=config.gamma_dephasing,
    )
    jax.config.update("jax_default_device", jax.devices(os.getenv("DEFAULT_DEVICE_SAMPLE_CIRC", "cpu"))[0])

    # %%
    print(f"Initializing sensor n={n}, k={k}")
    sensor = Sensor(n, k, **kwargs)

    # %%
    hf = h5py.File(io.path.joinpath("circ.h5"), "r")
    theta = jnp.array(hf.get("theta"))
    mu = jnp.array(hf.get("mu"))
    hf.close()

    # %% testing samples
    print(f"Sampling {n_shots_test} shots for {phis_test}.")
    t0 = time.time()
    shots_test, probs_test = sensor.sample_over_phases(
        theta, phis_test, mu, n_shots=n_shots_test, verbose=True, key=key
    )
    t1 = time.time()
    print(f"Sampling took {t1 - t0} seconds.")

    # %%
    hf = h5py.File(io.path.joinpath("test_samples.h5"), "w")
    hf.create_dataset("probs_test", data=probs_test)
    hf.create_dataset("shots_test", data=shots_test)
    hf.create_dataset("phis_test", data=phis_test)
    hf.close()

    # %%
    print(f"Finished sampling the circuits for test data.")

    return


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="tmp")
    args = parser.parse_args()
    folder = args.folder

    io = IO(folder=f"{folder}")
    print(io)
    config = Configuration.from_yaml(io.path.joinpath("config.yaml"))
    key = jax.random.PRNGKey(config.seed)
    print(
        f"Sampling circuit for testing data: {folder} | Devices {jax.devices()} | Full path {io.path}"
    )
    print(f"Config: {config}")
    sample_circuit_testing(io, config, key, progress=True, plot=True)
