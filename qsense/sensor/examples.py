import uuid

from qsense.sensor.unitaries import H, Identity, CNOT, U3, Phase
from qsense.sensor.blocks import Probe


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


def local_entangling_probe(n=2, d=2, n_layers=1):
    """

    :param n:
    :param d:
    :return:
    """
    # probe = list()
    probe = Probe(n=n)

    for layer in range(n_layers):
        # circuit.append([RDX(str(uuid.uuid4()), d=d) for _ in range(n)])
        probe.add([U3(str(uuid.uuid4()), d=d) for _ in range(n)])

        probe.add([CNOT(d=d, n=2, control=0, target=1) for _ in range(0, n // 2)])
        if n % 2 == 1:
            probe._circuit[-1].append(Identity(d=d))

        probe.add(
            [Identity(d=d)]
            + [CNOT(d=d, n=2, control=0, target=1) for i in range(1, (n + 1) // 2)]
        )
        if n % 2 == 0:
            probe._circuit[-1].append(Identity(d=d))

    return probe


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


def local_unitary_circuit(n=2, d=2):
    """

    :param n:
    :param d:
    :return:
    """
    circuit = list()
    circuit.append([U3(str(uuid.uuid4())) for _ in range(n)])
    return circuit


def local_phase_interaction_circuit(n=2, d=2):
    """

    :param n:
    :param d:
    :return:
    """
    circuit = list()
    circuit.append([Phase("phase", d=d) for _ in range(n)])
    return circuit
