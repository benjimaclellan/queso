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

import argparse

import jax

from queso.io import IO
from queso.configs import Configuration
from queso.train.train_circuit import train_circuit
from queso.sample.circuit import sample_circuit
from queso.sample.circuit_test import sample_circuit_testing
from queso.train.train_nn import train_nn
from queso.benchmark.estimator import benchmark_estimator


#%%
def vqs(io: IO, config: Configuration):
    """
    Executes the Variational Quantum Sensing (VQS) workflow.

    This function performs a series of operations based on the provided configuration.
    It can train a quantum circuit, sample from the circuit, train a neural network, and benchmark an estimator.
    Each operation is optional and controlled by the configuration.

    Args:
        io (IO): An instance of the IO class for handling input/output operations.
        config (Configuration): An instance of the Configuration class containing the settings for the VQS workflow.

    Returns:
        None
    """
    # %%
    key = jax.random.PRNGKey(config.seed)
    progress = True
    plot = True

    # %%
    _, key = jax.random.split(key)
    if config.train_circuit:
        train_circuit(
            io=io,
            config=config,
            key=key,
            progress=progress,
            plot=plot,
        )

    # %% sample circuit settings
    # _, key = jax.random.split(key)
    # if config.sample_circuit:
    _, key = jax.random.split(key)
    if config.sample_circuit_training_data:
        sample_circuit(
            io=io,
            config=config,
            key=key,
            plot=plot,
        )

    # %% sample circuit settings
    _, key = jax.random.split(key)
    if config.sample_circuit_testing_data:
        sample_circuit_testing(
            io=io,
            config=config,
            key=key,
            plot=plot,
        )

    # %% train estimator settings
    _, key = jax.random.split(key)
    if config.train_nn:
        train_nn(
            io=io,
            config=config,
            key=key,
            plot=plot,
            progress=progress,
        )

    # %% benchmark estimator
    _, key = jax.random.split(key)
    if config.benchmark_estimator:
        benchmark_estimator(
            io=io,
            config=config,
            key=key,
            plot=True,
        )
    return


#%%
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="tmp")
    args = parser.parse_args()
    folder = args.folder

    io = IO(folder=f"{folder}")
    print(io)
    config = Configuration.from_yaml(io.path.joinpath('config.yaml'))
    print(f"Initializing sensor training: {folder} | Devices {jax.devices()} | Full path {io.path}")
    print(f"Config: {config}")
    vqs(io, config)
