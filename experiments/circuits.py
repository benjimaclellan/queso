import uuid
from qsense.gates import *


def ghz_circuit(n=2, d=2):
    """
    Returns the canonical example of a GHZ quantum circuit for an n-partite, d-dimensional system.

    :param n:
    :param d:
    :return:
    """
    circuit = list()
    circuit.append([H() if i == 0 else Identity() for i in range(n)])
    for i in range(1, n):
        circuit.append([CNOT(n=n, control=0, target=i)])

    return circuit


def local_entangling_circuit(n=2, d=2, n_layers=1):
    """

    :param n:
    :param d:
    :return:
    """
    circuit = list()
    for layer in range(n_layers):
        circuit.append([U2(str(uuid.uuid4())) for _ in range(n)])
        circuit.append([CNOT(n=2, control=0, target=1) for i in range(1, n, 2)])
        circuit.append(
            [Identity()]
            + [CNOT(n=2, control=0, target=1) for i in range(2, n - 1, 2)]
            + [Identity()]
        )

    return circuit


def nonlocal_entangling_circuit(n=2, d=2):
    """

    :param n:
    :param d:
    :return:
    """
    circuit = list()
    circuit.append([U3(str(uuid.uuid4())) for _ in range(n)])
    for i in range(1, n):
        circuit.append([CNOT(n=n, control=0, target=i)])

    return circuit
