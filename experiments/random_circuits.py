from functools import partial
import time

import pandas as pd
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import tensorcircuit as tc
from tensorcircuit.quantum import sample_bin2int, sample_int2bin
import jax
import jax.numpy as jnp

backend = tc.set_backend("jax")
tc.set_dtype("complex128")
tc.set_contractor("greedy")  # “auto”, “greedy”, “branch”, “plain”, “tng”, “custom”

#%%
class Sensor:
    def __init__(
        self,
        n,
        k,
        **kwargs,
    ):
        self.n = n
        self.k = k
        self._circ = tc.Circuit
        # self._circ = tc.DMCircuit

        self.preparation = two_qubit_random_circuit
        # self.preparation = monitored_circuit
        self.theta = jnp.zeros([n, k, 3])

        self.interaction = local_rx
        self.phi = jnp.array(0.0)
        return

    def circuit(self, theta, phi, mu):
        c = self._circ(self.n)
        c = self.preparation(c, theta, self.n, self.k)
        c = self.interaction(c, phi, self.n)
        return c

    # @partial(jax.jit, static_argnums=(0,))
    def state(self, theta, phi):
        c = self._circ(self.n)
        c = self.preparation(c, theta, self.n, self.k)
        c = self.interaction(c, phi, self.n)
        return c.state()

    # @partial(jax.jit, static_argnums=(0,))
    def probs(self, theta, phi, mu):
        c = self._circ(self.n)
        c = self.preparation(c, theta, self.n, self.k)
        c = self.interaction(c, phi, self.n)
        # c = self.detection(c, mu, self.n, self.k)
        return c.probability()

    @partial(jax.jit, static_argnums=(0,), backend='cpu')
    def _sample(self, theta, phi, mu, key):
        c = self._circ(self.n)
        c = self.preparation(c, theta, self.n, self.k)
        c = self.interaction(c, phi, self.n)
        c = self.detection(c, mu, self.n, self.k)

        backend.set_random_state(key)
        return c.measure(*list(range(self.n)))[0]

    # @partial(jax.jit, static_argnums=(0,))
    def sample(self, theta, phi, mu, key=None, n_shots=100, verbose=False):
        if key is None:
            key = jax.random.PRNGKey(time.time_ns())
        keys = jax.random.split(key, n_shots)
        shots = jnp.array([self._sample(theta, phi, mu, key) for key in keys]).astype(
            "bool"
        )
        return shots

    # @partial(jax.jit, static_argnums=(0,))
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

    # @partial(jax.jit, static_argnums=(0,))
    def entanglement(self, theta, phi):
        state = self.state(theta, phi)
        rho_A = tc.quantum.reduced_density_matrix(
            state, [i for i in range(self.n // 2)]
        )
        entropy = tc.quantum.entropy(rho_A)
        return entropy

    def sample_over_phases(self, theta, phis, mu, n_shots, key=None, verbose=False):
        check = self.sample(theta, 0.0, mu, key=key, n_shots=1, verbose=verbose)
        print(f"Sampling at φ = {phis}")
        print(check.device())
        if key is None:
            key = jax.random.PRNGKey(time.time_ns())
        keys = jax.random.split(key, phis.shape[0])
        data = [
                self.sample(theta, phi, mu, key=key, n_shots=n_shots, verbose=verbose)
                for (phi, key) in tqdm(zip(phis, keys), total=phis.size)
        ]
        data = jnp.stack(data, axis=0)
        probs = jnp.stack([self.probs(theta, phi, mu) for phi in phis], axis=0)
        return data, probs


def two_qubit_random_circuit(c, theta, n, k):
    for j in range(k):
        for i in range(0, n-1, 2):
            # c.unitary(i, i+1, unitary=theta[i][j])
            c.unitary(i, i+1, unitary=tc.gates.Gate(tc.gates.random_two_qubit_gate()))

        for i in range(1, n-1, 2):
            # c.unitary(i, i+1, unitary=theta[i][j])
            c.unitary(i, i+1, unitary=tc.gates.Gate(tc.gates.random_two_qubit_gate()))

# def monitored_circuit(c, theta, n, k):
    # for j in range(k):
        # for i in range(n):
        #     c.r(
        #         i,
        #         theta=theta[i, j, 0],
        #         alpha=theta[i, j, 1],
        #         phi=theta[i, j, 2],
        #     )
        #
        # for i in range(0, n - 1, 2):
        #     c.cz(
        #         i,
        #         i + 1,
        #     )
        #
        # for i in range(1, n - 1, 2):
        #     c.cz(
        #         i,
        #         i + 1,
        #     )
    c.barrier_instruction()
    return c


# interaction
def local_rx(c, phi, n):
    for i in range(n):
        c.ry(i, theta=phi)
    c.barrier_instruction()
    return c


if __name__ == "__main__":

    #%%
    n = 8
    ks = jnp.arange(1, 10).tolist()

    n_trials = 50
    seed = jax.random.PRNGKey(0)

    keys = jax.random.split(seed, len(ks))
    phi = jnp.array(0.0)

    #%%
    data = []
    for k, key in zip(ks, keys):
        print(f"k={k}")
        sensor = Sensor(n, k)
        # theta = 2 * jnp.pi * jax.random.uniform(key=key, shape=sensor.theta.shape)
        for j, subkey in enumerate(jax.random.split(key, n_trials)):
            theta = 2 * jnp.pi * jax.random.uniform(key=subkey, shape=sensor.theta.shape)
            # theta = [[tc.gates.random_two_qubit_gate() for i in range(n)] for j in range(k)]

            fi = sensor.qfi(theta, phi)
            vn = sensor.entanglement(theta, phi)
            data.append(dict(k=k, j=j, n=n, fi=fi, vn=vn))

    data = pd.DataFrame(data)

    #%%

    fig, axs = plt.subplots(nrows=2)
    axs[0].scatter(data.k, data.fi, alpha=0.1)
    axs[0].axhline(n, ls='--', alpha=0.5)
    axs[1].scatter(data.k, data.vn, alpha=0.1)
    axs[-1].set(xlabel='Circuit depth')
    axs[0].set(ylabel='Quantum Fisher Information')
    axs[1].set(ylabel='Entropy of entanglement')
    fig.show()

    #%%