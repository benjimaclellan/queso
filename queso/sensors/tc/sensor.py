from functools import partial
import time
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

import tensorcircuit as tc
from tensorcircuit.quantum import sample_bin2int, sample_int2bin
import jax
import jax.numpy as jnp


backend = tc.set_backend("jax")
tc.set_dtype("complex128")

tc.set_contractor("auto")  # “auto”, “greedy”, “branch”, “plain”, “tng”, “custom”


class Sensor:
    def __init__(
        self,
        n,
        k,
        contractor="auto",
        backend="ket",
        preparation="cr_brick_wall",
        interaction="rx",
        detection="local",
    ):
        self.n = n
        self.k = k

        if backend == "ket":
            self._circ = tc.Circuit #(self.n)
        elif backend == "dm":
            self._circ = tc.DMCircuit #(self.n)
        else:
            raise ValueError

        # tc.set_contractor(contractor)  # “auto”, “greedy”, “branch”, “plain”, “tng”, “custom”

        if preparation == "cr_brick_wall":
            self.preparation = preparation_cr_brick_wall
            self.theta = jnp.zeros([n, k, 6])
        else:
            raise ValueError("Not a valid preparation layer.")

        if interaction == "rx":
            self.interaction = interaction_rx
            self.phi = jnp.array(0.0)
        else:
            raise ValueError("Not a valid interaction layer.")

        if detection == "local":
            self.detection = detection_local
            self.mu = jnp.zeros([n, 3])
        else:
            raise ValueError("Not a valid detection layer.")

        return

    def circuit(self, theta, phi, mu):
        c = self._circ(self.n)
        c = self.preparation(c, theta, self.n, self.k)
        c = self.interaction(c, phi, self.n)
        c = self.detection(c, mu, self.n, self.k)
        return c

    @partial(jax.jit, static_argnums=(0,))
    def state(self, theta, phi):
        c = self._circ(self.n)
        c = self.preparation(c, theta, self.n, self.k)
        c = self.interaction(c, phi, self.n)
        return c.state()

    @partial(jax.jit, static_argnums=(0,))
    def probs(self, theta, phi, mu):
        c = self._circ(self.n)
        c = self.preparation(c, theta, self.n, self.k)
        c = self.interaction(c, phi, self.n)
        c = self.detection(c, mu, self.n, self.k)
        return c.probability()

    @partial(jax.jit, static_argnums=(0,))
    def _sample(self, theta, phi, mu, key):
        c = self._circ(self.n)
        c = self.preparation(c, theta, self.n, self.k)
        c = self.interaction(c, phi, self.n)
        c = self.detection(c, mu, self.n, self.k)

        backend.set_random_state(key)
        return c.measure(*list(range(self.n)))[0]

    # @partial(jax.jit, static_argnums=(0,))
    def sample(self, theta, phi, mu, key=None, n_shots=100, verbose=False):
        print(f"Sampling {phi}")
        if key is None:
            key = jax.random.PRNGKey(time.time_ns())
        keys = jax.random.split(key, n_shots)
        shots = jnp.array([self._sample(theta, phi, mu, key) for key in keys]).astype(
            "bool"
        )
        return shots

    @partial(jax.jit, static_argnums=(0,))
    def qfi(self, theta, phi):
        psi = self.state(theta, phi)
        dpsi = jax.jacrev(self.state, argnums=1, holomorphic=True)(
            theta.astype("complex64"), phi.astype("complex64")
        )
        fi = (
            4
            * jnp.real(
                (
                    jnp.conj(dpsi[None, :]) @ dpsi[:, None]
                    - jnp.abs(jnp.conj(dpsi[None, :]) @ psi[:, None])
                )
            ).squeeze()
        )
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
        rho_A = tc.quantum.reduced_density_matrix(
            state, [i for i in range(self.n // 2)]
        )
        entropy = tc.quantum.entropy(rho_A)
        return entropy

    def sample_over_phases(self, theta, phis, mu, n_shots, key=None, verbose=False):
        if key is None:
            key = jax.random.PRNGKey(time.time_ns())
        keys = jax.random.split(key, phis.shape[0])
        data = jnp.stack(
            [
                self.sample(theta, phi, mu, key=key, n_shots=n_shots, verbose=verbose)
                for (phi, key) in zip(phis, keys)
            ],
            axis=0,
        )
        probs = jnp.stack([self.probs(theta, phi, mu) for phi in phis], axis=0)
        return data, probs


# preparation layers
def preparation_cr_brick_wall(c, theta, n, k):
    for j in range(k):
        for i in range(n):
            c.r(
                i,
                theta=theta[i, j, 0],
                alpha=theta[i, j, 1],
                phi=theta[i, j, 2],
            )

        for i in range(0, n - 1, 2):
            # c.cnot(i, i + 1)
            c.cr(
                i,
                i + 1,
                theta=theta[i, j, 3],
                alpha=theta[i, j, 4],
                phi=theta[i, j, 5],
            )

        for i in range(1, n - 1, 2):
            # c.cnot(i, i + 1)
            c.cr(
                i,
                i + 1,
                theta=theta[i, j, 3],
                alpha=theta[i, j, 4],
                phi=theta[i, j, 5],
            )
    return c


# interaction layers
def interaction_rx(c, phi, n):
    for i in range(n):
        c.rx(i, theta=phi)
    return c


# detection layers
def detection_local(c, mu, n, k):
    for i in range(n):
        c.r(
            i,
            theta=mu[i, 0],
            alpha=mu[i, 1],
            phi=mu[i, 2],
        )
    return c


# utilities for sampling
def shots_to_counts(shots):
    # shots = jnp.array(list(zip(*shots))[0]).astype("int8")
    basis, count = jnp.unique(shots, return_counts=True, axis=0)
    return {
        "".join([str(j) for j in basis[i]]): count[i].item() for i in range(len(count))
    }


def counts_to_list(counts):
    bin_str = ["".join(p) for p in itertools.product("01", repeat=n)]
    return [counts.get(b, 0) for b in bin_str]
