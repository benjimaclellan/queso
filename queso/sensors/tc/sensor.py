# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright (c) 2022-2024 Benjamin MacLellan

from functools import partial
import time
from tqdm import tqdm
import tensorcircuit as tc
import jax
import jax.numpy as jnp

from queso.sensors.tc.detection import *
from queso.sensors.tc.interaction import *
from queso.sensors.tc.preparation import *

backend = tc.set_backend("jax")
tc.set_dtype("complex128")
tc.set_contractor("auto")  # “auto”, “greedy”, “branch”, “plain”, “tng”, “custom”


class Sensor:
    """
    The Sensor class represents a quantum sensor. It is initialized with the number of qubits (n) and the number of layers (k) in the quantum circuit.

    The class provides methods for creating the quantum circuit, calculating the quantum state, probabilities, and quantum Fisher information (QFI), sampling measurements, and more.

    Attributes:
        n (int): The number of qubits in the quantum circuit.
        k (int): The number of layers in the quantum circuit.
        preparation (function): The function to prepare the quantum state.
        interaction (function): The function to apply the interaction Hamiltonian.
        detection (function): The function to apply the detection Hamiltonian.
        theta (jax.numpy.ndarray): The parameters for the preparation function.
        phi (float): The parameter for the interaction function.
        mu (jax.numpy.ndarray): The parameters for the detection function.
        layers (dict): A dictionary containing the names of the preparation, interaction, and detection layers.

    Methods:
        circuit(theta, phi, mu): Returns the quantum circuit.
        state(theta, phi): Returns the quantum state.
        probs(theta, phi, mu): Returns the probabilities of the quantum state.
        sample(theta, phi, mu, key=None, n_shots=100, verbose=False): Returns samples of measurements.
        qfi(theta, phi): Returns the quantum Fisher information.
        cfi(theta, phi, mu): Returns the classical Fisher information.
        entanglement(theta, phi): Returns the entanglement entropy.
        sample_over_phases(theta, phis, mu, n_shots, key=None, verbose=False): Returns samples of measurements over different phases.
    """
    def __init__(
        self,
        n,
        k,
        **kwargs,
    ):
        self.n = n
        self.k = k
        self.kwargs = kwargs
        backend = kwargs.get("backend", "ket")
        if backend == "ket":
            self._circ = tc.Circuit
        elif backend == "dm":
            self._circ = tc.DMCircuit
        else:
            raise ValueError

        # tc.set_contractor(contractor)  # “auto”, “greedy”, “branch”, “plain”, “tng”, “custom”

        # default circuits
        preparation = kwargs.get("preparation", "hardware_efficient_ansatz")
        interaction = kwargs.get("interaction", "local_rz")
        detection = kwargs.get("detection", "local_r")

        self.preparation, self.theta = set_preparation(preparation, n, k, kwargs)
        self.interaction, self.phi = set_interaction(interaction)
        self.detection, self.mu = set_detection(detection, n, k)
        self.layers = dict(
            preparation=preparation, interaction=interaction, detection=detection
        )

        return

    def init_params(self, key=None):
        if key is None:
            key = jax.random.PRNGKey(time.time_ns())
        keys = jax.random.split(key, 3)
        return (
            jax.random.uniform(keys[0], self.theta.shape),
            jax.random.uniform(keys[0], self.phi.shape),
            jax.random.uniform(keys[0], self.mu.shape),
        )

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

    @partial(jax.jit, static_argnums=(0,), backend="cpu")
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
        # fi = jnp.sum((jnp.power(dpr, 2) / pr))
        fi = jnp.nansum((jnp.power(dpr, 2) / pr))  # todo: check if removing nans helps/hurts numerical stability
        return fi

    @partial(jax.jit, static_argnums=(0,))
    def entanglement(self, theta, phi):
        # state = self.state(theta, phi)
        c = self._circ(self.n)
        c = self.preparation(c, theta, self.n, self.k)
        state = c.state()
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


def set_preparation(preparation, n, k, kwargs):
    """
    Sets the preparation layer for the quantum circuit based on the provided parameters.

    Args:
        preparation (str): The name of the preparation layer to be used.
        n (int): The number of qubits in the quantum circuit.
        k (int): The number of layers in the quantum circuit.
        kwargs (dict): Additional arguments for specific preparation layers.

    Returns:
        function: The function representing the preparation layer.
        jax.numpy.ndarray: The parameters for the preparation function.

    Raises:
        ValueError: If the provided preparation layer name is not valid.
    """

    if preparation == "hardware_efficient_ansatz":
        return hardware_efficient_ansatz, jnp.zeros([n, k + 1, 2])

    elif preparation == "hardware_efficient_ansatz_dephasing":
        gamma = kwargs.get("gamma_dephasing")
        return (
            lambda c, theta, n, k: hardware_efficient_ansatz_dephasing(c, theta, n, k, gamma=gamma),
            jnp.zeros([n, k + 1, 2])
        )

    elif preparation == "ghz_local_rotation_dephasing":
        gamma_dephasing = kwargs.get("gamma_dephasing")
        return (
            lambda c, theta, n, k: ghz_local_rotation_dephasing(c, theta, n, k, gamma=gamma_dephasing),
            jnp.zeros([n, 2, 2])
        )

    elif preparation == "ghz_dephasing":
        gamma_dephasing = kwargs.get("gamma_dephasing")
        return (
            lambda c, theta, n, k: ghz_dephasing(c, theta, n, k, gamma=gamma_dephasing),
            jnp.array([])
        )

    elif preparation == "trapped_ion_ansatz":
        return trapped_ion_ansatz, jnp.zeros([n, k + 1, 4])

    elif preparation == "photonic_graph_state_ansatz":
        # graph_state = kwargs.get("graph_state")
        return (
            photonic_graph_state_ansatz,
            jnp.zeros([n, 1, 3]),
        )

    elif preparation == "brick_wall_cr":
        return brick_wall_cr, jnp.zeros([n, k, 6])

    elif preparation == "brick_wall_cr_ancillas":
        n_ancilla = kwargs.get("n_ancilla", n // 2)
        return (
            lambda c, theta, n, k: brick_wall_cr_ancillas(
                c, theta, n, k, n_ancilla=n_ancilla
            ),
            jnp.zeros([n, k, 6]),
        )
    elif preparation == "brick_wall_rx_ry_cnot":
        return brick_wall_rx_ry_cnot, jnp.zeros([n, k, 3])

    elif preparation == "brick_wall_cr_dephasing":
        gamma_dephasing = kwargs.get("gamma_dephasing", 0.0)
        return (
            lambda c, theta, n, k: brick_wall_cr_dephasing(
                c, theta, n, k, gamma=gamma_dephasing
            ),
            jnp.zeros([n, k, 6]),
        )

    elif preparation == "brick_wall_cr_depolarizing":
        gamma_dephasing = kwargs.get("gamma_dephasing", 0.0)
        return (
            lambda c, theta, n, k: brick_wall_cr_depolarizing(
                c, theta, n, k, gamma=gamma_dephasing
            ),
            jnp.zeros([n, k, 6]),
        )

    elif preparation == "local_r":
        return local_r, jnp.zeros([n, 3])

    else:
        raise ValueError("Not a valid preparation layer.")


def set_interaction(interaction):
    phi = jnp.array(0.0)

    if interaction == "local_rx":
        return local_rx, phi
    if interaction == "local_rz":
        return local_rz, phi
    elif interaction == "single_rx":
        return single_rx, phi
    elif interaction == "fourier_rx":
        return fourier_rx, phi
    else:
        raise ValueError("Not a valid interaction layer.")


def set_detection(detection, n, k):
    if detection == "local_r":
        return local_r, jnp.zeros([n, 3])

    elif detection == "computational_bases":
        return computational_bases, jnp.array([])

    elif detection == "hadamard_bases":
        return hadamard_bases, jnp.array([])

    elif detection == "brick_wall_cr":
        return brick_wall_cr, jnp.zeros([n, k, 6])

    elif detection == "local_rx_ry_ry":
        return local_rx_ry_ry, jnp.zeros([n, 3])
    else:
        raise ValueError("Not a valid detection layer.")
