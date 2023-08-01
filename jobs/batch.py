import copy
import os
import sys
import pathlib
import subprocess
from math import pi

# path = "/home/projects/def-rgmelko/bmaclell/queso"
path = "/home/bmaclell/projects/def-rgmelko/bmaclell/queso"
sys.path.append(path)
# print(sys.path)

from queso.io import IO
from queso.configs import Configuration
from queso.train import train


base = Configuration()

folders = {}
for n in (6,):
    config = copy.deepcopy(base)
    config.n = n
    config.k = n
    config.train_circuit = False
    config.sample_circuit_training_data = False
    config.sample_circuit_testing_data = False
    config.n_shots_test = 10000
    config.train_nn = True
    config.benchmark_estimator = True
    config.phi_range = [-pi/n/2, pi/n/2]
    config.phis_test = [0.1 * pi/n/2]
    
    config.n_epochs = 10000
    config.batch_size = 50
        
    folder = f"2023-07-31_n{config.n}_k{config.k}"
    jobname = f"n{config.n}k{config.k}"

    io = IO(folder=folder)
    io.path.mkdir(parents=True, exist_ok=True)
    config.to_yaml(io.path.joinpath('config.yaml'))
    print(io.path.name)
    
    # train(io, config)

    # Use subprocess to call the sbatch command with the batch script, parameters, and Slurm time argument
    subprocess.run([
        # "pwd"
        "sbatch", 
        "--time=0:120:00", 
        "--account=def-rgmelko",
        "--mem=8000",
        f"--gpus-per-node=1",
        f"--job-name={jobname}.job",
        f"--output=out/{jobname}.out",
        f"--error=out/{jobname}.err",
        "submit.sh", str(folder)
        ]
    )
