import time

import jax
from jax import numpy as np
from qsense.utils import tensor, prod


def nketz0(n):
    return tensor(n * [ketz0()])


def nketx0(n):
    return tensor(n * [ketx0()])


def nket_ghz(n):
    return (1 / np.sqrt(2)) * (
        tensor(n * [np.array([[1.0, 0.0]]).T]) + tensor(n * [np.array([[0.0, 1.0]]).T])
    )


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


def cnot(n=2, control=0, target=1):
    d0 = {control: ketz0() @ ketz0().T, target: eye()}
    d1 = {control: ketz1() @ ketz1().T, target: x()}
    return tensor([d0.get(reg, eye()) for reg in range(n)]) + tensor(
        [d1.get(reg, eye()) for reg in range(n)]
    )


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


def initialize(circuit):
    params = {}
    rng_key = jax.random.PRNGKey(time.time_ns())
    for layer in circuit:
        for u in layer:
            if u.bounds is not None and u.m is not None:
                if u.key not in params.keys():
                    rng_key, rng_subkey = jax.random.split(rng_key)
                    params[u.key] = jax.random.uniform(
                        key=rng_subkey,
                        shape=[u.m],
                        minval=u.bounds[0],
                        maxval=u.bounds[1],
                    )
    return params


def compile(params, circuit):
    us = []
    for layer in circuit:
        us.append(
            tensor(
                [u(*params[u.key]) if (u.key in params.keys()) else u() for u in layer]
            )
        )
    u = prod(us)
    return u
