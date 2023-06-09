from functools import partial
import time
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

import tensorcircuit as tc
import jax
import jax.numpy as jnp


backend = tc.set_backend("jax")
tc.set_dtype("complex128")
tc.set_contractor("auto")  # “auto”, “greedy”, “branch”, “plain”, “tng”, “custom”


class Sensor:
    def __init__(self, n, k, shots=100):
        self.n = n
        self.k = k
        self.shots = shots
        return

    def preparation(self, c, theta):
        for j in range(self.k):
            for i in range(self.n):
                c.r(
                    i,
                    theta=theta[i, j, 0],
                    alpha=theta[i, j, 1],
                    phi=theta[i, j, 2],
                )

            for i in range(0, self.n-1, 2):
                c.cnot(i, i+1)

            for i in range(1, self.n-1, 2):
                c.cnot(i, i+1)
        return c

    def interaction(self, c, phi):
        for i in range(self.n):
            c.rx(i, theta=phi)
        return c

    def detection(self, c, mu):
        for i in range(self.n):
            c.r(
                i,
                theta=mu[i, 0],
                alpha=mu[i, 1],
                phi=mu[i, 2],
            )
        return c

    def circuit(self, theta, phi, mu):
        c = tc.Circuit(self.n)
        c = self.preparation(c, theta)
        c = self.interaction(c, phi)
        c = self.detection(c, mu)
        return c

    @partial(jax.jit, static_argnums=(0,))
    def state(self, theta, phi):
        c = tc.Circuit(self.n)
        c = self.preparation(c, theta)
        c = self.interaction(c, phi)
        return c.state()

    @partial(jax.jit, static_argnums=(0,))
    def probs(self, theta, phi, mu):
        c = tc.Circuit(self.n)
        c = self.preparation(c, theta)
        c = self.interaction(c, phi)
        c = self.detection(c, mu)
        return c.probability()

    @partial(jax.jit, static_argnums=(0,))
    def _sample(self, theta, phi, mu, key):
        c = tc.Circuit(self.n)
        c = self.preparation(c, theta)
        c = self.interaction(c, phi)
        c = self.detection(c, mu)

        backend.set_random_state(key)
        return c.measure(*list(range(self.n)))[0]

    # @partial(jax.jit, static_argnums=(0,))
    def sample(self, theta, phi, mu, key=None, n_shots=100):
        if key is None:
            key = jax.random.PRNGKey(time.time_ns())
        keys = jax.random.split(key, n_shots)
        shots = jnp.array([self._sample(theta, phi, mu, key) for key in keys]).astype("int8")
        return shots

    @partial(jax.jit, static_argnums=(0,))
    def qfi(self, theta, phi):
        psi = self.state(theta, phi)
        dpsi = jax.jacrev(self.state, argnums=1, holomorphic=True)(theta.astype("complex64"), phi.astype("complex64"))
        fi = 4 * jnp.real((jnp.conj(dpsi[None, :]) @ dpsi[:, None] - jnp.abs(jnp.conj(dpsi[None, :]) @ psi[:, None]))).squeeze()
        return fi

    @partial(jax.jit, static_argnums=(0,))
    def cfi(self, theta, phi, mu):
        pr = self.probs(theta, phi, mu)
        dpr = jax.jacrev(self.probs, argnums=1, holomorphic=False)(theta, phi, mu)
        fi = jnp.sum((jnp.power(dpr, 2) / pr))
        return fi

    @partial(jax.jit, static_argnums=(0,))
    def entanglement(self, theta, phi):
        state = self.state(theta, phi)
        rho_A = tc.quantum.reduced_density_matrix(state, [i for i in range(self.n//2)])
        entropy = tc.quantum.entropy(rho_A)
        return entropy

    def sample_over_phases(self, theta, phis, mu, n_shots, key=None):
        if key is None:
            key = jax.random.PRNGKey(time.time_ns())
        keys = jax.random.split(key, phis.shape[0])
        data = jnp.stack(
            [self.sample(theta, phi, mu, key=key, n_shots=n_shots) for (phi, key) in zip(phis, keys)],
            axis=0
        )
        return data


def shots_to_counts(shots):
    # shots = jnp.array(list(zip(*shots))[0]).astype("int8")
    basis, count = jnp.unique(shots, return_counts=True, axis=0)
    return {''.join([str(j) for j in basis[i]]): count[i].item() for i in range(len(count))}


def counts_to_list(counts):
    bin_str = [''.join(p) for p in itertools.product('01', repeat=n)]
    return [counts.get(b, 0) for b in bin_str]
