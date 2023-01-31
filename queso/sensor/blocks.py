"""
Blocks which form a quantum sensing/
"""

import jax
import time
from abc import ABC
from functools import reduce, partial
import jax.numpy as np

from queso.sensor.functions import tensor, prod, nketz0


class BlockBase(ABC):
    def __init__(self, n: int, d: int):
        self.n = n
        self.d = d
        self._circuit = []
        self._params = None
        return

    def initialize(self, seed=None):
        params = {}
        rng_key = jax.random.PRNGKey(time.time_ns() if seed is None else seed)
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

    def __call__(self, params: dict, method="greedy"):
        """
        Calculates the unitary matrix of a parameterized circuit block.

        :param params: dictionary of parameters, which are distributed to each gate model
        :return:
        """
        print("Method is ", method)
        if method == "greedy":
            us = []
            # todo: lazy generation of unitaries to reduce memory overhead
            for x in self._circuit:
                us.append(
                    tensor(
                        [
                            u(*params[u.key]) if (u.key in params.keys()) else u()
                            for u in x
                        ]
                    )
                )
            unitary = prod(reversed(us))

        elif method == "lazy":
            def _layer(_x):
                matrices = map(lambda u: u(*params[u.key]) if (u.key in params.keys()) else u(), _x)
                _u = reduce(np.kron, matrices)
                return _u

            layers = map(_layer, reversed(self._circuit))
            unitary = reduce(np.matmul, layers)

        else:
            raise NotImplemented("Not a valid method for compiling circuit unitary.")

        return unitary


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


class Sensor:
    def __init__(
        self,
        probe: Probe,
        interaction: Interaction = None,
        measurement: Measurement = None,
    ):
        self.probe = probe
        self.interaction = interaction
        self.measurement = measurement

        self.state_i = nketz0(n=self.probe.n, d=self.probe.d)

    def initialize(self, seed=None):
        params = {}
        for block in [self.probe, self.interaction, self.measurement]:
            params.update(block.initialize(seed=seed))
        return params

    def __call__(self, params, method="lazy"):
        # todo: improved compiler of sensor
        return (
            self.measurement(params, method=method)
            @ self.interaction(params, method=method)
            @ self.probe(params, method=method)
            @ self.state_i
        )
