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
from queso.sensors.tc.utils import sample_bin2int, sample_int2bin
from queso.io import IO
from queso.configs import Configuration


# %%
def sample_circuit(
    io: IO,
    config: Configuration,
    key: jax.random.PRNGKey,
    plot: bool = False,
    progress: bool = True,
):
    """
    Samples a quantum circuit based on the provided configuration and random key.

    This function initializes a sensor with the given configuration, samples the circuit,
    and optionally plots the results. The sampled data is saved in an HDF5 file.

    Args:
        io (IO): An IO object for handling file operations.
        config (Configuration): A Configuration object containing the parameters for the circuit.
        key (jax.random.PRNGKey): A random key for JAX operations.
        plot (bool, optional): If True, plots the true probabilities and relative frequencies. Defaults to False.
        progress (bool, optional): If True, displays progress information. Defaults to True.

    Returns:
        None
    """
    jax.config.update("jax_default_device", jax.devices(os.getenv("DEFAULT_DEVICE_SAMPLE_CIRC", "cpu"))[0])


    n = config.n
    k = config.k
    phi_range = config.phi_range
    n_phis = config.n_phis
    n_shots = config.n_shots
    kwargs = dict(
        preparation=config.preparation,
        interaction=config.interaction,
        detection=config.detection,
        backend=config.backend,
        n_ancilla=config.n_ancilla,
        gamma_dephasing=config.gamma_dephasing,
    )

    # %%
    print(f"Initializing sensor n={n}, k={k}")
    sensor = Sensor(n, k, **kwargs)

    # %%
    hf = h5py.File(io.path.joinpath("circ.h5"), "r")
    # print(hf.keys())
    theta = jnp.array(hf.get("theta"))
    mu = jnp.array(hf.get("mu"))
    hf.close()

    # %% training data set
    print(
        f"Sampling {n_shots} shots for {n_phis} phase value between {phi_range[0]} and {phi_range[1]}."
    )
    phis = (phi_range[1] - phi_range[0]) * jnp.arange(n_phis) / (
        n_phis - 1
    ) + phi_range[0]
    t0 = time.time()
    shots, probs = sensor.sample_over_phases(
        theta, phis, mu, n_shots=n_shots, verbose=True, key=key
    )
    t1 = time.time()
    print(f"Sampling took {t1 - t0} seconds.")

    # %%
    outcomes = sample_bin2int(shots, n)
    counts = jnp.stack(
        [
            jnp.count_nonzero(outcomes == x, axis=(1,), keepdims=True).squeeze()
            for x in range(2**n)
        ],
        axis=1,
    )
    freqs = counts / counts.sum(axis=-1, keepdims=True)
    bit_strings = sample_int2bin(jnp.arange(2**n), n)

    # %%
    if plot:
        # %%
        fig, axs = plt.subplots(nrows=2)
        sns.heatmap(probs, ax=axs[0], cbar_kws={"label": "True Probs."})
        sns.heatmap(freqs, ax=axs[1], cbar_kws={"label": "Rel. Freqs."})
        plt.show()
        io.save_figure(fig, filename="probs_freqs.png")

        colors = sns.color_palette("deep", n_colors=bit_strings.shape[0])
        fig, ax = plt.subplots()
        for i in range(bit_strings.shape[0]):
            xdata = jnp.linspace(
                phi_range[0], phi_range[1], probs.shape[0], endpoint=False
            )
            ax.plot(xdata, freqs[:, i], color=colors[i], ls="--", alpha=0.3)
        io.save_figure(fig, filename="liklihoods.png")

    # %%
    hf = h5py.File(io.path.joinpath("train_samples.h5"), "w")
    hf.create_dataset("probs", data=probs)
    hf.create_dataset("shots", data=shots)
    hf.create_dataset("counts", data=counts)
    hf.create_dataset("phis", data=phis)
    hf.close()

    print(f"Finished sampling the circuits.")
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
    print(f"Sampling circuit: {folder} | Devices {jax.devices()} | Full path {io.path}")
    print(f"Config: {config}")
    sample_circuit(io, config, key, progress=True, plot=True)
