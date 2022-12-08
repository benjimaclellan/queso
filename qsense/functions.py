import time
import itertools

import jax
from jax import numpy as np
from qsense.utils import tensor, prod


def basis(k, d):
    ket = np.zeros([d, 1])
    return ket.at[k].set(1.0)


def dagger(array):
    return array.conjugate().T


def ketz0(d=2):
    return basis(0, d)


def ketz1(d=2):
    return basis(1, d)


def ketx0(d=2):
    return 1 / np.sqrt(2) * np.array([[1.0], [1.0]])


def ketx1(d=2):
    return 1 / np.sqrt(2) * np.array([[1.0], [-1.0]])


def kety0(d=2):
    return 1 / np.sqrt(2) * np.array([[1.0], [1.0j]])


def kety1(d=2):
    return 1 / np.sqrt(2) * np.array([[1.0], [-1.0j]])


def nketz0(n, d=2):
    return tensor(n * [ketz0(d=d)])


def nketz1(n, d=2):
    return tensor(n * [ketz1(d=d)])


def nketx0(n, d=2):
    return tensor(n * [ketx0()])


def nketx1(n, d=2):
    return tensor(n * [ketx1()])


def nket_ghz(n, d=2):
    return (1 / np.sqrt(d)) * (
        sum([tensor(n * [basis(i, d).T]) for i in range(d)])
    )


def eye(d=2):
    return np.identity(d)


def x(d=2):
    return sum([basis((ell + 1) % d, d) @ dagger(basis(ell, d)) for ell in range(d)])


def y(d=2):
    return x(d) @ z(d)


def z(d=2):
    return sum(
        [
            np.exp(2j * np.pi * ell / d) * basis(ell, d) @ dagger(basis(ell, d))
            for ell in range(d)
        ]
    )


def h(d=2):
    return 1 / np.sqrt(d) * sum(
        [
            np.exp(2j * np.pi / d) ** (i * j) * basis(i, d) @ dagger(basis(j, d))
            for (i, j) in itertools.product(range(d), range(d))
        ]
    )


def cnot(d=2, n=2, control=0, target=1):
    ds = [
        {
            control: basis(i, d) @ dagger(basis(i, d)),
            target: sum(
                [basis((i + j) % d, d) @ dagger(basis(j, d)) for j in range(d)]
            ),
        }
        for i in range(d)
    ]
    u = sum([tensor([ds[i].get(reg, eye(d)) for reg in range(n)]) for i in range(d)])
    return u


def phase(phi, d=2):
    # return sum(
    #     [
    #         np.exp(1j * phi * ell) * basis(ell, d) @ dagger(basis(ell, d))
    #         for ell in range(d)
    #     ]
    # )
    return sum(
        [
            np.exp(1j * phi * ell) * basis(ell, d) @ dagger(basis(ell, d)) if ell == (d-1)
            else basis(ell, d) @ dagger(basis(ell, d))
            for ell in range(d)
        ]
    )
    # u = np.identity(d, dtype=np.complex128)
    # u.at[-1].set(np.exp(1j * phi))
    # return u


def rdx(*args, d=2):
    u = np.identity(d)
    for (i, j), param in zip(itertools.combinations(range(d), 2), args):
        rot = np.identity(d)
        rot = rot.at[i, j].set(-np.sin(param))
        rot = rot.at[j, i].set(np.sin(param))
        rot = rot.at[i, i].set(np.cos(param))
        rot = rot.at[j, j].set(np.cos(param))
        u = rot @ u
    return u


def rx(theta, d=2):
    return u3(theta, 0.0, 0.0)


def rz(phi, d=2):
    return np.array([[np.exp(-1j * phi / 2), 0.0], [0.0, np.exp(1j * phi / 2)]])


def u2(theta, phi, d=2):
    u = rx(theta, d=d) @ rz(phi, d=d)
    return u


def u3(theta, phi, lam, d=2):
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


def norm(ket):
    return np.sum(np.abs(ket) ** 2)


def is_norm(ket):
    return np.isclose(norm(ket), 1.0)


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

    u = prod(reversed(us))
    return u


def check(params, circuit):
    us = []
    for layer in circuit:
        u = tensor(
            [
                gate(*params[gate.key]) if (gate.key in params.keys()) else gate()
                for gate in layer
            ]
        )
        print(
            "total difference", np.sum(np.identity(u.shape[0]) - dagger(u) @ u), layer
        )
    u = prod(reversed(us))
    return u
