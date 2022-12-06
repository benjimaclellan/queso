import jax.numpy as np
from jax.numpy.linalg import eigh

from qsense.utils import tensor
from qsense.states import ketz0, ketz1


def dagger(array):
    return array.conjugate().T


def eye():
    return np.array([[1.0, 0.0], [0.0, 1.0]])


def x():
    return np.array([[0.0, 1.0], [1.0, 0.0]])


def y():
    return np.array([[0.0, -1.0j], [1.0j, 0.0]])


def z():
    return np.array([[1.0, 0.0], [0.0, -1.0]])


def h():
    return np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2)


def cnot():
    return tensor([ketz0() @ ketz0().T, eye()]) + tensor([ketz1() @ ketz1().T, x()])


def phase(phi):
    return np.array([[1.0, 0.0], [0.0, np.exp(1j * phi)]])


def rx(theta):
    return u3(theta, 0.0, 0.0)


def rz(phi):
    return np.array([[np.exp(-1j * phi / 2), 0.0], [0.0, np.exp(1j * phi / 2)]])


def u2(theta, phi):
    u = rx(theta) @ rz(phi)
    return u


def u3(theta, phi, lam):
    u = np.array(
        [
            [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
            [
                np.exp(1j * phi) * np.sin(theta / 2),
                np.exp(1j * (phi + lam)) * np.cos(theta / 2),
            ],
        ]
    )
    return u


def is_norm(ket):
    return np.isclose(np.sum(np.abs(ket) ** 2), 1.0)


def is_hermitian(input_matrix):
    return np.allclose(input_matrix, np.conjugate(input_matrix.T))


# def is_psd(input_matrix, perturbation=1e-15):
#     # first check if it is a Hermitian matrix
#     if not is_hermitian(input_matrix):
#         return False
#
#     perturbed_matrix = input_matrix + perturbation * np.identity(len(input_matrix))
#     try:
#         np.linalg.cholesky(perturbed_matrix)
#         return True
#     except np.linalg.LinAlgError:
#         # np.linalg.cholesky throws this exception if the matrix is not positive definite
#         return False


# def is_density_matrix(input_matrix, perturbation=1e-15):
#     return is_psd(input_matrix, perturbation) and np.allclose(input_matrix.trace(), 1.0)
#
#
# def is_pure(rho):
#     return np.allclose(np.real(np.trace(rho @ rho)), 1.0)

"""
m: number of parameters
bounds: (low, high) bounds that the parameter must fall within
initial: starting parameter vector, if not sampled
"""

unitary_info = {
    phase: {"m": 1, "bounds": (-np.pi, np.pi), "initial": [0.0]},
    rx: {"m": 1, "bounds": (-np.pi, np.pi), "initial": [0.0]},
    rz: {"m": 1, "bounds": (-np.pi, np.pi), "initial": [0.0]},
    u2: {"m": 2, "bounds": (-np.pi, np.pi), "initial": [0.0, 0.0]},
    u3: {"m": 3, "bounds": (-np.pi, np.pi), "initial": [0.0, 0.0, 0.0]},
}

# gates with no parameters
unitary_info.update(
    {func: {"m": 0, "bounds": (), "initial": []} for func in (x, y, z, h, eye, cnot)}
)
