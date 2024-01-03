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
