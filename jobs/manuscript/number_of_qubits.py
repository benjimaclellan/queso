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
jax.config.update("jax_default_device", jax.devices(os.getenv("DEFAULT_DEVICE", "cpu"))[0])

from queso.io import IO
from queso.train.vqs import vqs
from queso.configs import Configuration#


folders = {}
ansatz = "hardware_efficient_ansatz"
ns = [
    # 2,
    # 4,
    # 6,
    # 8,
    10
]
phi_centers = [np.pi/2/n for n in ns]
seeds = [
    # 0,
    # 0,
    # 0,
    # 0,
    123
]

for (n, seed, phi_center) in zip(ns, seeds, phi_centers):
    print(n, ansatz)
    config = Configuration()
    config.preparation = ansatz
    config.n = n
    config.k = n

    prefix = f"{config.preparation}"
    folder = f"number_of_qubits/n{config.n}"

    config.train_circuit = True
    config.sample_circuit_training_data = True
    config.sample_circuit_testing_data = True
    config.train_nn = True
    config.benchmark_estimator = True

    config.seed = seed
    config.n_grid = 250

    config.interaction = 'local_rx'
    config.detection = 'local_r'
    config.loss_fi = "loss_cfi"

    config.n_shots = 1000
    config.n_shots_test = 10000
    config.n_phis = 250
    config.phi_center = phi_center
    config.phi_range = [-pi/2/n + config.phi_center, pi/2/n + config.phi_center]

    config.phis_test = np.linspace(-pi/3/n + config.phi_center, pi/3/n + config.phi_center, 5).tolist()
    config.n_sequences = np.logspace(0, 3, 10, dtype='int').tolist()
    config.n_epochs = 2000
    config.lr_nn = 1.0e-3
    config.l2_regularization = 0.001

    # config.n_grid = 500
    config.nn_dims = [64, 64, 64]
    config.batch_size = 1000

    jobname = f"{prefix}n{config.n}k{config.k}"
    io = IO(path=data_path, folder=folder, include_date=True)

    if os.getenv("CLUSTER", "false") == "false":
        io.save_yaml(config, 'config.yaml')
        vqs(io, config)

    else:
        io.save_yaml(config, 'config.yaml')
        # Use subprocess to call the sbatch command with the batch script, parameters, and Slurm time argument
        subprocess.run([
            # "pwd"
            "sbatch",
            "--time=0:30:00",
            "--account=def-rgmelko",
            "--mem=4000",
            # f"--gpus-per-node=1",
            f"--job-name={jobname}.job",
            f"--output=out/{jobname}.out",
            f"--error=out/{jobname}.err",
            "submit.sh", str(folder)
        ]
        )