import copy
import itertools
import os
import sys
import pathlib
import subprocess
from math import pi
import platform
import numpy as np
import networkx as nx


current_os = platform.system()
if current_os == "Linux":
    module_path = "/home/bmaclell/projects/def-rgmelko/bmaclell/queso"
    data_path = "/home/bmaclell/projects/def-rgmelko/bmaclell/queso/data"

elif current_os == "Darwin":
    from queso.io import IO
    from queso.train.vqs import vqs

    module_path = "/"
    data_path = "/Users/benjamin/data/queso"
else:
    raise EnvironmentError
sys.path.append(module_path)


from queso.configs import Configuration


base = Configuration()

folders = {}

ansatzes = [
    # "hardware_efficient_ansatz",
    # "trapped_ion_ansatz",
    'photonic_graph_state_ansatz',
]
ns = [4, ]

for (ansatz, n) in itertools.product(ansatzes, ns):
    print(n, ansatz)
    config = copy.deepcopy(base)
    config.preparation = ansatz

    # if ansatz == "photonic_graph_state_ansatz":
    #     # use a bipartite graph (optimal QFI, see Shettel et al.)
    #     graph = nx.Graph()
    #     nodes_a, nodes_b = [i for i in range(0, n // 2)], [i for i in range(n//2, n)]
    #     graph.add_nodes_from(nodes_a, bipartite=0)
    #     graph.add_nodes_from(nodes_b, bipartite=1)
    #     graph.add_edges_from(list(itertools.product(nodes_a, nodes_b)))
    #
    #     config.graph_state = graph

    config.n = n
    config.k = n
    config.n_grid = 250
    
    config.seed = 122344
    
    prefix = f"{config.preparation}"
    folder = f"2024-01-08_hardware_ansatzes/n{config.n}_k{config.k}_{config.preparation}"

    config.train_circuit = False
    config.sample_circuit_training_data = False
    config.sample_circuit_testing_data = False
    config.train_nn = True
    config.benchmark_estimator = True

    config.interaction = 'local_rx'
    config.detection = 'local_r'
    config.loss_fi = "loss_cfi"
    
    config.n_shots = 1000
    config.n_shots_test = 10000
    config.n_phis = 250
    config.phi_range = [-pi/2/n, pi/2/n]

    config.phis_test = np.linspace(-pi/3/n, pi/3/n, 5).tolist()  # [-0.4 * pi, -0.1 * pi, -0.5 * pi/n/2]
    config.n_sequences = np.logspace(0, 3, 10, dtype='int').tolist()
    config.n_epochs = 1000
    config.lr_nn = 0.5e-4
    config.l2_regularization = 0.1
    
    # config.n_grid = 500
    config.nn_dims = [64, 64, 64]
    config.batch_size = 1000
    
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
        vqs(io, config)


