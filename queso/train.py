import argparse

import time
import jax
import jax.numpy as jnp

from queso.io import IO
from queso.configs import Configuration
from queso.train_circuit import train_circuit
from queso.sample_circuit import sample_circuit
from queso.train_nn import train_nn
from queso.benchmark_estimator import benchmark_estimator


#%%
def train(io: IO, config: Configuration):
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
    _, key = jax.random.split(key)
    if config.sample_circuit:
        sample_circuit(
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
            plot=False,
        )
    return


#%%
if __name__ == "__main__":

    #%%
    config = Configuration()
    config.folder = "2023-07-27_test_pipe"
    # config.train_circuit = False
    # config.sample_circuit = False
    config.n_epochs = 10
    config.n_steps = 1000

    #%%
    io = IO(folder=f"{config.folder}")
    io.path.mkdir(parents=True, exist_ok=True)

    config.to_yaml(io.path.joinpath('config.yaml'))

    # %%
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--folder", type=str, default="nonlocal-det-single-int")
    # parser.add_argument("--n", type=int, default=2)
    # parser.add_argument("--k", type=int, default=2)
    # parser.add_argument("--seed", type=int, default=None)
    # args = parser.parse_args()

    # folder = args.folder
    # n = args.n
    # k = args.k
    # seed = args.seed if args.seed is not None else time.time_ns()

    train(io, config)