import copy
import os
import sys
import pathlib
import subprocess
from math import pi
import platform
import numpy as np
 

current_os = platform.system()
if current_os == "Linux":
    module_path = "/home/bmaclell/projects/def-rgmelko/bmaclell/queso"
    data_path = "/home/bmaclell/projects/def-rgmelko/bmaclell/queso/data"

elif current_os == "Darwin":
    from queso.io import IO
    from queso.train import train

    module_path = "/Users/benjamin/Library/CloudStorage/OneDrive-UniversityofWaterloo/Desktop/1 - Projects/Quantum Intelligence Lab/queso"
    data_path = "/Users/benjamin/data/queso"
else:
    raise EnvironmentError
sys.path.append(module_path)


from queso.configs import Configuration


base = Configuration()

folders = {}
n = 2
config = copy.deepcopy(base)
config.n = n
config.k = n
config.seed = 123

folder = f"2023-09-12_ibm_qasm_test"

config.train_circuit = True
config.sample_circuit_training_data = False
config.sample_circuit_testing_data = False
config.train_nn = False
config.benchmark_estimator = False

config.preparation = 'brick_wall_rx_ry_cnot'
config.interaction = 'local_rx'
config.detection = 'local_rx_ry_ry'
config.n_ancilla = 1
config.loss_fi = "loss_cfi"
config.backend = "dm"

config.n_shots = 1000
config.n_shots_test = 10000
config.n_phis = 200
config.phi_range = [-pi/2/n, pi/2/n]

config.phis_test = np.linspace(-pi/3/n, pi/3/n, 6).tolist()  # [-0.4 * pi, -0.1 * pi, -0.5 * pi/n/2]
config.n_sequences = np.logspace(0, 3, 10, dtype='int').tolist()
config.n_epochs = 10000
config.lr_nn = 1e-4
config.l2_regularization = 1.0
config.n_grid = 200
config.nn_dims = [64, 64]
config.batch_size = 250


if current_os == "Linux":
    jobname = f"{prefix}n{config.n}k{config.k}"

    path = pathlib.Path(data_path).joinpath(folder)
    path.mkdir(parents=True, exist_ok=True)
    config.to_yaml(path.joinpath('config.yaml'))
    print(path.name)
    # Use subprocess to call the sbatch command with the batch script, parameters, and Slurm time argument
    subprocess.run([
        # "pwd"
        "sbatch",
        "--time=0:60:00",
        "--account=def-rgmelko",
        "--mem=8000",
        f"--gpus-per-node=1",
        f"--job-name={jobname}.job",
        f"--output=out/{jobname}.out",
        f"--error=out/{jobname}.err",
        "submit.sh", str(folder)
        ]
    )
elif current_os == "Darwin":
    io = IO(path=data_path, folder=folder)
    train(io, config)


