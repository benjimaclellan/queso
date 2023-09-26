import copy
import os
import sys
import pathlib
import subprocess
from math import pi
import platform
import numpy as np
import shutil


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



prefix = f"ibm_q"
folder = f"2023-09-19_ibmq_ibmq_manila_n4"

path = pathlib.Path(data_path).joinpath(folder)
config = Configuration.from_yaml(path.joinpath('config.yaml'))
# config.seed = 122344

config.train_circuit = False
config.sample_circuit_training_data = False
config.sample_circuit_testing_data = False
config.train_nn = True
config.benchmark_estimator = True

config.n_sequences = np.logspace(0, 3, 10, dtype='int').tolist()
config.n_epochs = 30000
config.lr_nn = 0.1e-4
config.l2_regularization = 0.1

# config.n_grid = config.n_phis
config.nn_dims = [64, 64, 64]
config.batch_size = 500

jobname = f"{prefix}n{config.n}k{config.k}"

if current_os == "Linux":
    path = pathlib.Path(data_path).joinpath(folder)
    path.mkdir(parents=True, exist_ok=True)
    config.to_yaml(path.joinpath('config.yaml'))
    print(path.name)
    # Use subprocess to call the sbatch command with the batch script, parameters, and Slurm time argument
    subprocess.run([
        # "pwd"
        "sbatch",
        "--time=0:20:00",
        "--account=def-rgmelko",
        # "--mem=16000",
        "--mem=4000",
        f"--gpus-per-node=1",
        f"--job-name={jobname}.job",
        # "--constraint=cascade,v100",  # for high-memory GPUs
        f"--output=out/{jobname}.out",
        f"--error=out/{jobname}.err",
        "submit.sh", str(folder)
        ]
    )
elif current_os == "Darwin":
    io = IO(path=data_path, folder=folder)
    train(io, config)


