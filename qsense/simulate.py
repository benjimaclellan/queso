import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import uuid
import functools

from qsense.unitaries import *
from qsense.states import *
from qsense.utils import tensor, sum, prod

import numpy


def initialize(circuit):
    params = {}
    for layer in circuit:
        for (u, key) in layer:
            if (unitary_info[u]["bounds"] is not None) and (unitary_info[u]["m"] != 0):
                m = unitary_info[u]["m"]
                (low, high) = unitary_info[u]["bounds"]
                if key not in params.keys():
                    params[key] = np.complex128(numpy.random.uniform(low, high, m))
    return params


@jax.jit
def compile(params, circuit):
    us = []
    for layer in circuit:
        us.append(
            tensor(
                [
                    u(*params[key]) if (key in params.keys()) else u()
                    for (u, key) in layer
                ]
            )
        )
    u = prod(us)
    return u
