import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import uuid
from functools import partial
import seaborn as sns
import optax

from qsense.gates import *
from qsense.functions import *
from qsense.utils import tensor, sum, prod
from qsense.functions import initialize, compile
from qsense.qfi import qfim

from examples.circuits import *


if __name__ == "__main__":
    n = 4  # number of particles

    ket_i = nketz0(n)
    # ket_i = nket_ghz(n)

    # circuit = ghz_circuit(n)
    circuit = local_entangling_circuit(n)
    circuit.append([Phase("phase") for i in range(n)])

    params = initialize(circuit)
    params['phase'] = np.array([0.0])

    compile = jax.jit(partial(compile, circuit=circuit))
    # compile = partial(compile, circuit=circuit)
    u = compile(params)
    ket_f = u @ ket_i
    # print(ket_f)
    # print(norm(ket_f))

    keys = ["phase"]
    qfi = lambda params, circuit, ket_i, keys: qfim(params, circuit, ket_i, keys)[0, 0]
    qfi = jax.jit(partial(qfi, circuit=circuit, ket_i=ket_i, keys=keys))
    # qfi = partial(qfi, circuit=circuit, ket_i=ket_i, keys=keys)

    ell = qfi(params)
    print(ell)

    for _ in range(100):
        params = initialize(circuit)
        ell = qfi(params)
        print(ell, params)
