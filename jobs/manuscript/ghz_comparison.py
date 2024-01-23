from rich import print as pprint
import os
import sys
import pathlib
from math import pi
import numpy as np
from dotenv import load_dotenv
import jax
import jax.numpy as jnp

env_path = pathlib.Path(__file__).parent.parent.parent.joinpath('paths.env')
load_dotenv(env_path)
sys.path.append(os.getenv("MODULE_PATH"))
data_path = os.getenv("DATA_PATH")
jax.config.update("jax_default_device", jax.devices(os.getenv("DEFAULT_DEVICE", "cpu"))[0])


from queso.io import IO
from queso.train.vqs import vqs
from queso.configs import Configuration
from queso.benchmark.ghz import ghz_comparison

n = 4
config_ghz = Configuration(
    n=n,
    preparation="ghz_dephasing",
    interaction="local_rz",
    detection="hadamard_bases",
    seed=123,
)
config_vqs = Configuration(
    n=n, k=1,
    preparation="ghz_local_rotation_dephasing",
    interaction="local_rz",
    detection="local_r",
    seed=123,
)
# gammas = jnp.logspace(-3.5, -0.5, 8)
gammas = [
    0.001
]

#%%
for i, gamma in enumerate(gammas):
    ios = []
    for config in (
            config_ghz,
            config_vqs
    ):
        #%%
        config.backend = "dm"
        # folder = f"ghz_comparison/n{n}/{config.preparation}_gamma{gamma}"
        folder = f"ghz_comparison/n{n}/{config.preparation}_gamma_{i}"
        io = IO(path=data_path, folder=folder)
        config.phi_center = np.pi / 2 / n

        config.train_circuit = True
        config.sample_circuit_training_data = False
        config.sample_circuit_testing_data = False
        config.train_nn = False
        config.benchmark_estimator = False

        # config.n_steps = 10
        config.gamma_dephasing = gamma

        config.metrics = ['entropy_vn', 'ghz_fidelity']
        config.phi_range = [-pi/2/n + config.phi_center, pi/2/n + config.phi_center]

        io.save_yaml(config, filename='config.yaml')

        print(io.path)
        ios.append(io)

    ghz_comparison(ios[0], config_ghz)
    vqs(ios[1], config_vqs)
