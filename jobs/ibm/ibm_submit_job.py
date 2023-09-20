
#%%
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.test.reference_circuits import ReferenceCircuits
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, IBMBackend
from qiskit import QuantumCircuit, Aer, execute

from queso.io import IO

#%%
io = IO(folder="2023-09-12_ibm_qasm_test", include_id=False)
qasm = io.load_txt(filename='circ.qasm')
qc = QuantumCircuit.from_qasm_str(qasm)
# If you want to read from file, use instead
# qc = QuantumCircuit.from_qasm_file(io.path.joinpath("circ.qasm"))
# qc.measure_all()
qc.measure([0, 1], [0, 1])
qc.draw()
#%%

# backend = Aer.get_backend("qasm_simulator")
#
# # Execute the circuit and show the result.
# job = execute(qc, backend)
# result = job.result()
# print(result.get_counts())


#%%
service = QiskitRuntimeService(
    channel="ibm_cloud",
    token="739851cf054da7611b0226d31654b24eedbe349e142316e65308cac146b3f09a9bfc69d8b8d1c9b6f52314fd14c3da29db4e670d438ec953eed1fbef553fcc7a",
    # instance="ibm-q/open/main",
    # token="kWK3sBw15C98JNIx5i8XxojsRD4sdtlBA5m75Rzmfgbq",
    # instance="crn:v1:bluemix:public:quantum-computing:us-east:a/54c218cc3af54a698f2c91a2733a85c5:604438de-546e-4cf6-8172-56dccc817bc3::",
    # instance="crn:v1:bluemix:public:quantum-computing:us-east:a/54c218cc3af54a698f2c91a2733a85c5:b3033f6c-7e0b-421b-84cf-4a40265703c4::",
    # instance="crn:v1:bluemix:public:quantum-computing:us-east:a/54c218cc3af54a698f2c91a2733a85c5:bf5b213d-21ae-4033-9b0d-a15642ebdf69::",
)

#%%
service.backends()
# service.backends(simulator=False, operational=True)

#%%
backend = IBMBackend('ibm_algiers')

#%%
job = Sampler("ibmq_qasm_simulator").run(ReferenceCircuits.bell())

print(f"job id: {job.job_id()}")
result = job.result()
print(result)

#%%
# Prepare the input circuit.
#
# bell = QuantumCircuit(2)
# bell.h(0)
# bell.cx(0, 1)
# bell.measure_all()

#%%
# Execute the circuit
from qiskit_ibm_runtime import Sampler

backend = service.backend("ibm_algiers")
sampler = Sampler(backend)
job = sampler.run(circuits=qc)

#%%
job.status()

#%%
print(job.result())

#%%