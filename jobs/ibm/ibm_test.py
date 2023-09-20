

#%%
import copy
import os
import sys
import pathlib
import subprocess
from math import pi
import matplotlib.pyplot as plt
import seaborn as sns
import platform
import numpy as np
import h5py

from qiskit import execute, IBMQ
from qiskit import QuantumCircuit, Aer, execute

from queso.io import IO
from queso.train import train
from queso.configs import Configuration
from queso.sensors.tc.sensor import Sensor
from queso.sensors.tc.sensor import Sensor, sample_bin2int, sample_int2bin

module_path = "/Users/benjamin/Library/CloudStorage/OneDrive-UniversityofWaterloo/Desktop/1 - Projects/Quantum Intelligence Lab/queso"
data_path = "/Users/benjamin/data/queso"

def convert_results(results, phis):
    def hex_to_str(hex):
        return "{0:05b}".format(int(hex, 16))

    def str_to_array(string):
        return np.array(list(map(int, string)))

    shots = []
    for result, phi in zip(results, phis):  # loop through phi values, retrieving the
        print(
            result.header.name,
            result.shots,
            result.data.counts,
        )
        counts = result.data.counts

        tmp = [np.tile(str_to_array(hex_to_str(hx_val)), (count, 1)) for hx_val, count in counts.items()]
        shot_bits_per_basis = np.concatenate(tmp, axis=0)
        np.random.shuffle(shot_bits_per_basis)
        shots.append(shot_bits_per_basis)
    shots = np.stack(shots, axis=2)
    shots = np.swapaxes(shots, 0, 2)
    shots = np.swapaxes(shots, 1, 2)

    outcomes = sample_bin2int(shots, n)
    counts = np.stack([np.count_nonzero(outcomes == x, axis=(1,), keepdims=True).squeeze() for x in range(2 ** n)], axis=1)

    return shots, counts

#%%
folder = f"2023-09-19_ibmq"
io = IO(path=data_path, folder=folder)

#%%
n = 5
config = Configuration()
config.n = n
config.k = 4
config.seed = 1234

config.train_circuit = True
config.sample_circuit_training_data = False
config.sample_circuit_testing_data = False
config.train_nn = False
config.benchmark_estimator = False

config.preparation = 'brick_wall_rx_ry_cnot'
config.interaction = 'local_rx'
config.detection = 'local_rx_ry_ry'
config.loss_fi = "loss_cfi"

config.n_shots = 1000
config.n_shots_test = 10000
config.n_phis = 100
config.n_grid = 100
config.phi_range = [-pi/2/n, pi/2/n]
config.phis_test = np.linspace(-pi/3/n, pi/3/n, 6).tolist()  # [-0.4 * pi, -0.1 * pi, -0.5 * pi/n/2]

config.to_yaml(io.path.joinpath("config.yaml"))

#%%
train(io, config)

#%% make QASM files
n = config.n
k = config.k
phi_range = config.phi_range
n_phis = config.n_phis
n_shots = config.n_shots
kwargs = dict(preparation=config.preparation, interaction=config.interaction, detection=config.detection, backend=config.backend)

# %%
print(f"Initializing sensor n={n}, k={k}")
sensor = Sensor(n, k, **kwargs)

hf = h5py.File(io.path.joinpath("circ.h5"), "r")
# print(hf.keys())
theta = np.array(hf.get("theta"))
mu = np.array(hf.get("mu"))
phis = np.array(hf.get("phis"))
phis_test = config.phis_test
hf.close()

for phi in phis:
    qasm = sensor.circuit(theta, phi, mu).to_openqasm()
    io.save_txt(qasm, f"train/{phi}")

#%%
for phi in phis_test:
    qasm = sensor.circuit(theta, phi, mu).to_openqasm()
    io.save_txt(qasm, f"test/{phi}")

#%%
provider = IBMQ.load_account()

#%%
backend = provider.get_backend("ibmq_qasm_simulator")
# backend = provider.get_backend('ibm_lagos')
# backend = provider.get_backend('ibm_perth')

#%%
print("Beginning sampling from IBMQ device")
circuits_train = []
for phi in phis:
    qc = QuantumCircuit.from_qasm_file(io.path.joinpath(f"train/{phi}"))
    qc.measure([i for i in range(config.n)], [i for i in range(config.n)])
    circuits_train.append(qc)

#%%
job_train = execute(circuits_train, backend, shots=config.n_shots)
print(job_train.job_id())

#%%
job = backend.retrieve_job(job_train.job_id())
job.status()

#%%
results = job.result().results
shots, counts = convert_results(results, phis)

#%%
hf = h5py.File(io.path.joinpath("train_samples.h5"), "w")
hf.create_dataset("shots", data=shots)
hf.create_dataset("phis", data=phis)
hf.create_dataset("counts", data=counts)
hf.close()

print(f"Finished sampling the circuits.")
#%%
# Testing data
# %%
circuits_test = []
for phi in phis_test:
    qc = QuantumCircuit.from_qasm_file(io.path.joinpath(f"test/{phi}"))
    qc.measure([i for i in range(config.n)], [i for i in range(config.n)])
    circuits_test.append(qc)

# %%
job_test = execute(circuits_test, backend, shots=config.n_shots_test)
print(job_test.job_id())

# %%
job = backend.retrieve_job(job_test.job_id())
job.status()

# %%
results = job.result().results
shots_test, counts_test = convert_results(results, phis)

hf = h5py.File(io.path.joinpath("test_samples.h5"), "w")
hf.create_dataset("shots_test", data=shots_test)
hf.create_dataset("phis_test", data=phis_test)
# hf.create_dataset("counts_test", data=counts_test)
hf.close()

#%%
config.train_circuit = False
config.train_nn = True
config.benchmark_estimator = True

train(io, config)



#%%
outcomes = sample_bin2int(shots, n)
counts = np.stack([np.count_nonzero(outcomes == x, axis=(1,), keepdims=True).squeeze() for x in range(2 ** n)],
                   axis=1)
freqs = counts / counts.sum(axis=-1, keepdims=True)
bit_strings = sample_int2bin(np.arange(2 ** n), n)

fig, axs = plt.subplots(nrows=2)
# sns.heatmap(probs, ax=axs[0], cbar_kws={'label': 'True Probs.'})
sns.heatmap(freqs, ax=axs[1], cbar_kws={'label': 'Rel. Freqs.'})
plt.show()
# io.save_figure(fig, filename="probs_freqs.png")
