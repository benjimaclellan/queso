import copy
import itertools
import os
import sys
import pathlib
import subprocess
from math import pi
import platform
import numpy as np
from dotenv import load_dotenv
import pathlib
import jax

env_path = pathlib.Path(__file__).parent.parent.parent.joinpath('paths.env')
load_dotenv(env_path)
sys.path.append(os.getenv("MODULE_PATH"))
data_path = os.getenv("DATA_PATH")
# jax.config.update("jax_default_device", jax.devices(os.getenv("DEFAULT_DEVICE", "cpu"))[0])

from queso.io import IO
from queso.train.vqs import vqs
from queso.configs import Configuration


ansatze = [
    "hardware_efficient_ansatz",
]
ns = [4, ]
loss_funcs = [
    "loss_cfi",
    # "loss_qfi",
]

for (ansatz, n, loss_fi) in itertools.product(ansatze, ns, loss_funcs):
    print(n, ansatz)
    config = Configuration()
    config.preparation = ansatz

    prefix = f"{config.preparation}"
    folder = f"vqs-example-data/n{config.n}_{loss_fi}"

    config.train_circuit = False
    config.sample_circuit_training_data = False
    config.sample_circuit_testing_data = False
    config.train_nn = False
    config.benchmark_estimator = True

    config.n = n
    config.k = n
    config.n_grid = 250

    config.seed = 744

    config.interaction = 'local_rx'
    config.detection = 'local_r'
    config.loss_fi = loss_fi

    config.lr_circ = 0.5e-3
    config.n_shots = 1000
    config.n_shots_test = 10000
    config.n_phis = 250
    config.phi_center = pi/2/n
    config.phi_range = [-pi/2/n + config.phi_center, pi/2/n + config.phi_center]

    config.phis_test = np.linspace(-pi/3/n + config.phi_center, pi/3/n + config.phi_center, 5).tolist()  # [-0.4 * pi, -0.1 * pi, -0.5 * pi/n/2]
    config.n_sequences = np.logspace(0, 3, 10, dtype='int').tolist()
    config.n_epochs = 1000
    config.lr_nn = 0.5e-4
    config.l2_regularization = 0.1

    # config.n_grid = 500
    config.nn_dims = [64, 64, 64]
    config.batch_size = 1000

    io = IO(path=data_path, folder=folder)
    io.save_yaml(config, 'config.yaml')
    vqs(io, config)
