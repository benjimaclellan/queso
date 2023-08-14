import copy
import os
import sys
import pathlib
import subprocess
from math import pi
import platform
import time
import numpy as np


module_path = "/home/bmaclell/projects/def-rgmelko/bmaclell/queso"
data_path = "/home/bmaclell/projects/def-rgmelko/bmaclell/queso/data"
sys.path.append(module_path)

from queso.configs import Configuration


base = Configuration()

folders = {}
# account = "rrg-rgmelko-ab"
account = "def-rgmelko"

for n in (10, ):
    config = copy.deepcopy(base)
    config.n = n
    config.k = n
    folder = f"2023-08-02_nonlocal_prep_n{config.n}_k{config.k}"

    config.folder = folder
    config.train_circuit = False
    config.sample_circuit_training_data = False
    config.sample_circuit_testing_data = False
    config.train_nn = True
    config.benchmark_estimator = True

    config.preparation = 'brick_wall_cr_ancillas'
    # config.preparation = 'brick_wall_cr'
    # config.preparation = 'local_r'
    config.interaction = 'local_rx'
    config.detection = 'local_r'
    config.n_ancilla = 2

    config.n_shots = 5000
    config.n_shots_test = 10000

    config.phi_range = [-pi, pi]
    config.phis_test = (np.arange(-4, 5) / 10 * pi).tolist()  # [-0.4 * pi, -0.1 * pi, -0.5 * pi/n/2]
    config.n_sequences = np.logspace(0, 3, 10, dtype='int').tolist()
    config.n_epochs = 10000
    config.lr_nn = 5e-3
    config.nn_dims = [128, 128, 128]
    config.batch_size = 50
        
    jobname = f"n{config.n}k{config.k}"
    jobname_circ = f"{jobname}-circ"
    jobname_sample = f"{jobname}-sample"
    jobname_nn = f"{jobname}-nn"

    path = pathlib.Path(data_path).joinpath(folder)
    path.mkdir(parents=True, exist_ok=True)
    config.to_yaml(path.joinpath('config.yaml'))
    print(path.name)

    # SUBMIT CIRCUIT TRAINING
    flags = [False, False, True]
    
    if flags[0]:
        result = subprocess.run([
            "sbatch",
            f"--time=0:{n//2}0:00",
            f"--account={account}",
            "--mem=6000",
            f"--gpus-per-node=1",
            f"--job-name={jobname_circ}.job",
            f"--output=out/{jobname_circ}.out",
            f"--error=out/{jobname_circ}.err",
            "submit_circ.sh", str(folder)
            ],
            capture_output = True
        )
        print(result)
        job_id_circ = str(result.stdout.decode()).split()[-1]
    
    if flags[1]:
        # SUBMIT CIRCUIT SAMPLING
        opts = [
            f"--time=0:{n//2}0:00",
            f"--account={account}",
            "--mem=6000",
            # f"--gpus-per-node=1",
            f"--job-name={jobname_sample}.job",
            f"--output=out/{jobname_sample}.out",
            f"--error=out/{jobname_sample}.err",
        ]
        if flags[0]:
            opts.append(f"--dependency=afterok:{job_id_circ}")
            
        result = subprocess.run(["sbatch"] + opts + ["submit_nn.sh", str(folder)], capture_output = True)
        print(result)
        job_id_sample = str(result.stdout.decode()).split()[-1]
    
    if flags[2]:
        # SUBMIT NN TRAINING & BENCHMARKING
        opts = [
            "--time=0:100:00",
            f"--account={account}",
            "--mem=4000",
            f"--gpus-per-node=1",
            f"--job-name={jobname_nn}.job",
            f"--output=out/{jobname_nn}.out",
            f"--error=out/{jobname_nn}.err",
        ]
        if flags[0] and flags[1]:
            opts.append(f"--dependency=afterok:<{jobname_circ}:{jobname_sample}>")
            
        result = subprocess.run(["sbatch"] + opts + ["submit_nn.sh", str(folder)], capture_output = True)
        print(result)
        job_id_ = str(result.stdout.decode()).split()[-1]
        
