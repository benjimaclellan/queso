"""
Blocks which form a quantum sensing/
"""

import jax
import time
from abc import ABC

from qsense.sensor.functions import tensor, prod, nketz0


class BlockBase(ABC):
    def __init__(self, n: int, d: int):
        self.n = n
        self.d = d
        self._circuit = []
        self._params = None
        return

    def initialize(self):
        params = {}
        rng_key = jax.random.PRNGKey(time.time_ns())
        for layer in self._circuit:
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

    def add(self, layer):
        self._circuit.append(layer)

    def __call__(self, params: dict):
        """
        Calculates the unitary matrix of a parameterized circuit block.

        :param params: dictionary of parameters, which are distributed to each gate model
        :return:
        """
        us = []
        # todo: lazy generation of unitaries to reduce memory overhead
        for layer in self._circuit:
            us.append(
                tensor(
                    [
                        u(*params[u.key]) if (u.key in params.keys()) else u()
                        for u in layer
                    ]
                )
            )
        u = prod(reversed(us))
        return u


class Probe(BlockBase):
    """
    Base class for a probe block of the sensor
    """
    def __init__(self, n: int, d: int = 2):
        super().__init__(n, d)
        return


class Interaction(BlockBase):
    """
    Base class for the interaction block of the sensor
    """
    def __init__(self, n: int, d: int = 2):
        super().__init__(n, d)
        return


class Measurement(BlockBase):
    """
    Base class for the measurement block of the sensor
    """
    def __init__(self, n: int, d: int = 2):
        super().__init__(n, d)
        self._povm = []
        return


class Estimator(BlockBase):
    """
    Base class for the estimator block of the sensor
    """
    def __init__(self):
        super().__init__(n, d)
        return
