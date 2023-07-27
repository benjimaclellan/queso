import copy
import os
import sys
import pathlib
import subprocess
import subprocess

# path = "/home/projects/def-rgmelko/bmaclell/queso"
path = "/home/bmaclell/projects/def-rgmelko/bmaclell/queso"
sys.path.append(path)
# print(sys.path)

from queso.io import IO
from queso.configs import Configuration


base = Configuration()

folders = {}
for n in (2, 3, 4):
    config = copy.deepcopy(base)
    config.n = n
    config.train_circuit = False
    config.sample_circuit = False
        
    folder = f"2023-07-27_n={config.n}_k={config.k}"
    jobname = f"n{config.n}k{config.k}"

    io = IO(folder=folder)
    io.path.mkdir(parents=True, exist_ok=True)
    config.to_yaml(io.path.joinpath('config.yaml'))
    print(io.path.name)

    # Use subprocess to call the sbatch command with the batch script, parameters, and Slurm time argument
    subprocess.run([
        # "pwd"
        "sbatch", 
        "--time=0:20:00", 
        "--account=def-rgmelko",
        "--mem=4000",
        f"--gpus-per-node=1",
        f"--job-name={jobname}.job",
        f"--output=out/{jobname}.out",
        f"--error=out/{jobname}.err",
        "submit.sh", str(folder)
        ]
    )
