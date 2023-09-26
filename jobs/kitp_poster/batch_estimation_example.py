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
for n in (4,):
    
    config = copy.deepcopy(base)
    config.n = n
    config.k = n
    config.n_grid = 250
    
    config.seed = 122344
    
    prefix = f"est_ex_cfi_m{config.n_grid}"
    folder = f"2023-09-21_poster_estimation_example/n{config.n}_k{config.k}_m{config.n_grid}"

    config.train_circuit = True
    config.sample_circuit_training_data = True
    config.sample_circuit_testing_data = True
    config.train_nn = False
    config.benchmark_estimator = False

    config.preparation = 'brick_wall_cr'
    config.interaction = 'local_rx'
    config.detection = 'local_r'
    config.loss_fi = "loss_cfi"
    
    config.n_shots = 1000
    config.n_shots_test = 10000
    config.n_phis = 250
    config.phi_range = [-pi/2/n, pi/2/n]

    config.phis_test = np.linspace(-pi/3/n, pi/3/n, 5).tolist()  # [-0.4 * pi, -0.1 * pi, -0.5 * pi/n/2]
    config.n_sequences = np.logspace(0, 3, 10, dtype='int').tolist()
    config.n_epochs = 10000
    config.lr_nn = 0.5e-4
    config.l2_regularization = 0.1
    
    # config.n_grid = 500
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
    elif current_os == "Darwin":
        io = IO(path=data_path, folder=folder)
        train(io, config)


