import copy
import os
import sys
import pathlib
import subprocess
from math import pi
import platform
import numpy as np


module_path = "/home/bmaclell/projects/def-rgmelko/bmaclell/queso"
data_path = "/home/bmaclell/projects/def-rgmelko/bmaclell/queso/data"
sys.path.append(module_path)

from queso.configs import Configuration


base = Configuration()

folders = {}
for n in (4,):
    config = copy.deepcopy(base)
    config.n = n
    config.k = n
    folder = f"2023-08-02_nonlocal_prep_n{config.n}_k{config.k}"

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
    config.n_epochs = 1000
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
    subprocess.run([
        "sbatch",
        "--time=0:30:00",
        "--account=def-rgmelko",
        "--mem=4000",
        f"--gpus-per-node=1",
        f"--job-name={jobname_circ}.job",
        f"--output=out/{jobname_circ}.out",
        f"--error=out/{jobname_circ}.err",
        "submit_circ.sh", str(folder)
        ]
    )

    # SUBMIT CIRCUIT SAMPLING
    subprocess.run([
        "sbatch",
        "--time=0:30:00",
        "--account=def-rgmelko",
        "--mem=4000",
        # f"--gpus-per-node=1",
        f"--job-name={jobname_sample}.job",
        f"--output=out/{jobname_sample}-sample.out",
        f"--error=out/{jobname_sample}.err",
        f"--dependency=afterok:<{jobname_circ}>"
        "submit_sample.sh", str(folder)
    ]
    )

    # SUBMIT NN TRAINING & BENCHMARKING
    subprocess.run([
        "sbatch",
        "--time=0:80:00",
        "--account=def-rgmelko",
        "--mem=4000",
        f"--gpus-per-node=1",
        f"--job-name={jobname_nn}-nn.job",
        f"--output=out/{jobname_nn}-nn.out",
        f"--error=out/{jobname_nn}-nn.err",
        f"--dependency=afterok:<{jobname_circ}:{jobname_sample}>"
        "submit_nn.sh", str(folder)
    ]
    )