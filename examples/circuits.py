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

    circuit.append([Phase("phase") for _ in range(n)])
    return circuit
