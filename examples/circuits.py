import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import uuid
from functools import partial
import seaborn as sns
import optax

from qsense.unitaries import *
from qsense.states import *
from qsense.utils import tensor, sum, prod
from qsense.simulate import initialize, compile
from qsense.qfi import qfim


def ghz_circuit(n=2, d=2):
    ket_i = nketx0(n)

    circuit = []
    circuit.append([(h, str(uuid.uuid4())) for i in range(n)])
    circuit.append([(cnot, str(uuid.uuid4())) for i in range(0, n, 2)])
    circuit.append([(phase, "phase") for i in range(n)])

    params = initialize(circuit)

    # compile = jax.jit(partial(compile, circuit=circuit))
    compile = partial(compile, circuit=circuit)
    u = compile(params)

    # keys = ["phase"]
    # qfi = lambda params, circuit, ket_i, keys: qfim(params, circuit, ket_i, keys)[0, 0]
    # # qfi = jax.jit(partial(qfi, circuit=circuit, ket_i=ket_i, keys=keys))
    # qfi = partial(qfi, circuit=circuit, ket_i=ket_i, keys=keys)
    #
    # ell = qfi(params)
    # print(ell)
    #
    # for _ in range(10):
    #     params = initialize(circuit)
    #     print(params)
    #     u = compile(params)
    #     ell = qfi(params)
    #     print(ell)
    #
