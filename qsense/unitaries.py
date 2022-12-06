import jax.numpy as np
from jax.numpy.linalg import eigh

from qsense.utils import tensor
from qsense.states import ketz0, ketz1


def dagger(array):
    return array.conjugate().T






class FixedUnitary:
    def __init__(self):
        self.something = 1
        return


class ParameterizedUnitary:
    def __init__(self):
        self.something = 1
        return


class Identity(FixedUnitary):
    def __call__(self):
        return np.array([[1.0, 0.0], [0.0, 1.0]])


class X(FixedUnitary):
    def __call__(self):
        return np.array([[0.0, 1.0], [1.0, 0.0]])


class Y(FixedUnitary):
    def __call__(self):
        return np.array([[0.0, -1.0j], [1.0j, 0.0]])


class Z(FixedUnitary):
    def __call__(self):
        return np.array([[1.0, 0.0], [0.0, -1.0]])


class H(FixedUnitary):
    def __call__(self):
        return np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2)


class CNOT(FixedUnitary):
    def __init__(self, n=2, control=0, target=1):
        super().__init__()
        self.n = n
        self.control = control
        self.target = target
        self.eye = Identity()
        self.x = X()

    def __call__(self):
        d0 = {self.control: ketz0() @ ketz0().T, self.target: self.eye}
        d1 = {self.control: ketz1() @ ketz1().T, self.target: self.x}
        return tensor([d0.get(reg, self.eye) for reg in range(self.n)]) + tensor([d1.get(reg, self.eye) for reg in range(self.n)])


class Phase(FixedUnitary):
    def __call__(self, phi):
        return np.array([[1.0, 0.0], [0.0, np.exp(1j * phi)]])


class RX(FixedUnitary):
    def __call__(self, theta):
        return u3(theta, 0.0, 0.0)


class RZ(FixedUnitary):
    def __call__(self, phi):
        return np.array([[np.exp(-1j * phi / 2), 0.0], [0.0, np.exp(1j * phi / 2)]])


# class U2(FixedUnitary):
#     def __call__(self, theta, phi):
#         u = rx(theta) @ rz(phi)
#         return u
#
#
# def u3(theta, phi, lam):
#     u = np.array(
#         [
#             [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
#             [
#                 np.exp(1j * phi) * np.sin(theta / 2),
#                 np.exp(1j * (phi + lam)) * np.cos(theta / 2),
#             ],
#         ]
#     )
#     return u


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
    Phase: {"m": 1, "bounds": (-np.pi, np.pi), "initial": [0.0]},
    RX: {"m": 1, "bounds": (-np.pi, np.pi), "initial": [0.0]},
    RZ: {"m": 1, "bounds": (-np.pi, np.pi), "initial": [0.0]},
    # u2: {"m": 2, "bounds": (-np.pi, np.pi), "initial": [0.0, 0.0]},
    # u3: {"m": 3, "bounds": (-np.pi, np.pi), "initial": [0.0, 0.0, 0.0]},
}

# gates with no parameters
unitary_info.update(
    {func: {"m": 0, "bounds": (), "initial": []} for func in (X, Y, Z, CNOT, H, Identity)}
)
