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

# ns = (2, 3, 4, 5, 6, 7, 8)
# seeds = (2, 2, 2, 2, 3, 2, 2)
ns = (6,)
seeds = (12435,)

# ns = (9, 10)
# seeds = (2, 2)

for fi in ('qfi',): # 'cfi'):
    for ind, (n, seed) in enumerate(zip(ns, seeds)):
        prefix = f"{fi}_{n}"
        
        config = copy.deepcopy(base)
        config.n = n
        config.k = n
        config.seed = seed
        folder = f"2023-09-21_sweep_n_fi/{fi}_n{config.n}_k{config.k}"

        config.train_circuit = True
        config.sample_circuit_training_data = False
        config.sample_circuit_testing_data = False
        config.train_nn = False
        config.benchmark_estimator = False

        config.preparation = 'brick_wall_cr'
        config.interaction = 'local_rx'
        config.detection = 'local_r'
        config.loss_fi = f"loss_{fi}"
        config.backend = 'ket'
            
        jobname = f"{prefix}"

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
                "--account=rrg-rgmelko-ab",
                # "--account=def-rgmelko",
                # "--mem=12000",
                "--mem=4000",
                # f"--gpus-per-node=1",
                f"--job-name={jobname}.job",
                f"--output=out/{jobname}.out",
                f"--error=out/{jobname}.err",
                "submit.sh", str(folder)
                ]
            )
        elif current_os == "Darwin":
            io = IO(path=data_path, folder=folder)
            train(io, config)


