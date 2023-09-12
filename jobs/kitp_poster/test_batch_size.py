import copy
import os
import sys
import pathlib
import subprocess
from math import pi
import platform
import time
import numpy as np
import shutil
from pathlib import Path
 
 
def copy_files_to_new_folder(base_folder, new_folder):
    # Create a Path object for the base folder
    base_folder_path = Path(base_folder)

    # Ensure that the source folder exists
    if not base_folder.exists() or not base_folder.is_dir():
        print(f"The source folder '{base_folder}' does not exist or is not a directory.")
        return

    # Create a Path object for the new folder
    new_folder_path = Path(new_folder)

    # Create the new folder if it doesn't exist
    if not new_folder_path.exists():
        new_folder_path.mkdir(parents=True)

    # Iterate over all files in the base folder and copy them to the new folder
    for file in base_folder.glob('*'):
        if file.is_file():
            # Use shutil.copy to copy the file
            shutil.copy(str(file), str(new_folder_path / file.name))
            print(f"Copied '{file.name}' to '{new_folder}'")


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
n = 8

for batch_size in (10, 100, 250, 500, 1000):
    prefix = f"test_batch_{batch_size}"
    
    config = copy.deepcopy(base)
    config.n = n
    config.k = n
    config.seed = time.time_ns()
    config.l2_regularization = 0.1
    
    print(config)
    base_folder = f"2023-09-12_poster_estimation_example_n{config.n}_k{config.k}"
    folder = f"2023-09-12_characterize_batch_size_{batch_size}"
    
    base_dir = pathlib.Path(data_path).joinpath(base_folder)
    new_dir = pathlib.Path(data_path).joinpath(folder)
    
    if new_dir.exists():
        shutil.rmtree(new_dir)
    shutil.copytree(base_dir, new_dir)
    
    config.train_circuit = False
    config.sample_circuit_training_data = False
    config.sample_circuit_testing_data = False
    config.train_nn = True
    config.benchmark_estimator = True

    config.preparation = 'brick_wall_cr'
    config.interaction = 'local_rx'
    config.detection = 'local_r'
    config.loss_fi = "loss_cfi"
    
    config.n_shots = 1000
    config.n_shots_test = 10000
    config.n_phis = 200
    config.phi_range = [-pi/2/n, pi/2/n]

    config.phis_test = np.linspace(-pi/3/n, pi/3/n, 5).tolist()  # [-0.4 * pi, -0.1 * pi, -0.5 * pi/n/2]
    config.n_sequences = np.logspace(0, 3, 10, dtype='int').tolist()
    config.n_epochs = 20000
    config.lr_nn = 1e-4
    config.n_grid = 200
    config.nn_dims = [64, 64]
    config.batch_size = batch_size
    
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
            "--mem=3000",
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
