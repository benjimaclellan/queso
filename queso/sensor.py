from typing import Sequence
import itertools
import time
import tqdm
import matplotlib.pyplot as plt

import pennylane as qml
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn


class Sensor:

    def __init__(
        self,
        n,
        k,
        shots=1,
        probe="brick",
        interaction="z",
        detection="local",
    ):
        self.n = n
        self.k = k
        self.shots = shots

        self.probe = probe
        self.interaction = interaction
        self.detection = detection
        self.interface = "jax"
        self.device = qml.device("default.qubit", wires=n, shots=None)

        self.state = jax.jit(qml.QNode(self._state, device=self.device, interface=self.interface))
        self.probs = jax.jit(qml.QNode(self._probs, device=self.device, interface=self.interface))
        self.sample_nonjit = qml.QNode(self._sample, device=self.device, interface=self.interface)
        self.counts_nonjit = qml.QNode(self._counts, device=self.device, interface=self.interface)

        return

    def _preparation(self, theta):
        for j in range(self.k):
            for i in range(self.n):
                qml.RX(theta[i, 3 * j], wires=i)
                qml.RY(theta[i, 3 * j + 1], wires=i)
                qml.RZ(theta[i, 3 * j + 2], wires=i)
            for i in range(0, self.n - 1, 2):
                qml.CZ(wires=[i, i + 1])
            for i in range(1, self.n - 1, 2):
                qml.CZ(wires=[i, i + 1])
        return qml

    def _interaction(self, phi):
        for i in range(self.n):
            qml.RX(phi, wires=i)
        return qml

    def _detection(self, mu):
        for i in range(self.n):
            qml.RZ(mu[i, 0], wires=i)
            qml.RX(mu[i, 1], wires=i)
            qml.RY(mu[i, 2], wires=i)
        return qml

    def sensor(self, theta, phi, mu):
        self._preparation(theta)
        self._interaction(phi)
        self._detection(mu)
        return qml

    def _state(self, theta, phi, mu):
        return self.sensor(theta, phi, mu).state()

    def _probs(self, theta, phi, mu):
        return self.sensor(theta, phi, mu).probs()

    def _sample(self, theta, phi, mu):
        return self.sensor(theta, phi, mu).sample()

    def _counts(self, theta, phi, mu):
        return self.sensor(theta, phi, mu).counts()

    def sample(self, theta, phi, mu, shots, key=None):
        if key is None:
            key = jax.random.PRNGKey(time.time_ns())
        probs = self.probs(theta, phi, mu)
        inds = jax.random.choice(key, len(probs), shape=(shots,), replace=True, p=probs)
        bases = list(itertools.product(*self.n * [[0, 1]]))
        samples = jnp.array([bases[ind] for ind in inds])
        return samples

    def counts(self, theta, phi, mu, shots, key=None):
        # samples = self.sample(theta, phi, mu, shots, key=key)
        counts = {}
        # for k in range(shots):
        #     sample = tuple(samples[k, :].tolist())
        #     if tuple(sample) not in counts.keys():
        #         counts[sample] = 1
        #     else:
        #         counts[sample] += 1
        if key is None:
            key = jax.random.PRNGKey(time.time_ns())
        probs = self.probs(theta, phi, mu)
        unique, counts = jnp.unique(x, return_counts=True)
        inds = jax.random.choice(key, len(probs), shape=(shots,), replace=True, p=probs)
        bases = list(itertools.product(*self.n * [[0, 1]]))
        samples = jnp.array([bases[ind] for ind in inds])
        return counts

    def qfi(self, theta, phi, mu):
        psi = self.state(theta, phi, mu)
        dpsi = jax.jacrev(self.state, argnums=1, holomorphic=True)(theta.astype("complex64"), phi.astype("complex64"), mu.astype("complex64"))
        fi = 4 * jnp.real((jnp.conj(dpsi[None, :]) @ dpsi[:, None] - jnp.abs(jnp.conj(dpsi[None, :]) @ psi[:, None]))).squeeze()
        return fi

    def cfi(self, theta, phi, mu):
        pr = self.probs(theta, phi, mu)
        dpr = jax.jacrev(self.probs, argnums=1, holomorphic=False)(theta, phi, mu)
        fi = jnp.sum((jnp.power(dpr, 2) / pr))
        return fi


class RegressionEstimator(nn.Module):
    # todo: build FF-NN for regression
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for features in self.features[:-1]:
            x = nn.relu(nn.Dense(features=features)(x))
        x = nn.Dense(self.features[-1])(x)
        return x