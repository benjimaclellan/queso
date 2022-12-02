import jax.numpy as np
from jax.numpy.linalg import eigh

from qsense.utils import tensor


def nketz0(n):
    return tensor(n * [ketz0()])


def nketx0(n):
    return tensor(n * [ketx0()])


def nket_ghz(n):
    return (1/np.sqrt(2)) * (tensor(n * [np.array([[1.0, 0.0]]).T]) + tensor(n * [np.array([[0.0, 1.0]]).T]))


def ketx0():
    return 1 / np.sqrt(2) * np.array([[1.0], [1.0]])


def ketx1():
    return 1 / np.sqrt(2) * np.array([[1.0], [-1.0]])


def ketz0():
    return np.array([[1.0], [0.0]])


def ketz1():
    return np.array([[0.0], [1.0]])


def kety0():
    return 1 / np.sqrt(2) * np.array([[1.0], [1.0j]])


def kety1():
    return 1 / np.sqrt(2) * np.array([[1.0], [-1.0j]])
