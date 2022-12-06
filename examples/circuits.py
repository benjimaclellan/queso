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
    """
    Returns the canonical example of a GHZ quantum circuit for an n-partite, d-dimensional system.

    :param n:
    :param d:
    :return:
    """
    circuit = list()
    circuit.append([(h, None) if i == 0 else (eye, None) for i in range(n)])
    for i in range(1, n):
        circuit.append([(cnot(n=n, control=0, target=i), None)])

    circuit.append([(phase, "phase") for _ in range(n)])
    return circuit
