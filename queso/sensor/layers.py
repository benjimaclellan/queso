import uuid

from queso.sensor.unitaries import H, Identity, CNOT, U3, Phase
from queso.sensor.blocks import Probe


def ghz_probe(n=2, d=2):
    """
    Probe state circuit which prepares a GHZ state.

    :param n: number of qudits
    :param d: local dimension, d, of each local Hilbert space
    :return:
    """
    probe = list()
    probe.append([H() if i == 0 else Identity() for i in range(n)])
    for i in range(1, n):
        probe.append([CNOT(n=n, control=0, target=i)])

    return probe


def brick_wall_probe(n=2, d=2, n_layers=1):
    """
    Brick wall local entangling circuit, with two-local entangling gates and local rotations.

    :param n: number of qudits
    :param d: local dimension, d, of each local Hilbert space
    :return:
    """
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
    Probe state circuit composed of one control qudit connected to all others.

    :param n: number of qudits
    :param d: local dimension, d, of each local Hilbert space
    :return:
    """
    circuit = list()
    circuit.append([U3(str(uuid.uuid4())) for _ in range(n)])
    for i in range(1, n):
        circuit.append([CNOT(n=n, control=0, target=i)])

    return circuit


def local_rotations_circuit(n=2, d=2):
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
