import copy
from functools import partial

from queso.quantities.fischer_information import qfim, qfim_rho
from benchmarks.circuits import *


if __name__ == "__main__":
    n = 2  # number of particles
    d = 2  # local dimensions

    ket_i = nketz0(n, d=d)
    circuit = list()
    for layer in range(1):
        circuit.append([RDX(str(uuid.uuid4()), d=d) for _ in range(n)])
        circuit.append([CNOT(d=d, n=2, control=0, target=1) for _ in range(1, n, 2)])
    circuit.append([Phase("phase", d=d) for i in range(n)])

    params = initialize(circuit)
    params["phase"] = np.array([0.0])
    keys = ["phase"]

    # qfi = lambda params, circuit, ket_i, keys: qfim_rho(params, circuit, ket_i, keys)[0, 0]
    qfi = lambda params, circuit, ket_i, keys: qfim_rho(params, circuit, ket_i, keys)[
        0, 0
    ]
    # qfi = jax.jit(partial(qfi, circuit=circuit, ket_i=ket_i, keys=keys))
    qfi = partial(qfi, circuit=circuit, ket_i=ket_i, keys=keys)
    qq = qfi(copy.deepcopy(params))
    print(qq)

    qfi = lambda params, circuit, ket_i, keys: qfim(params, circuit, ket_i, keys)[0, 0]
    qfi = partial(qfi, circuit=circuit, ket_i=ket_i, keys=keys)
    qq = qfi(copy.deepcopy(params))
    print(qq)
    print("fin")
