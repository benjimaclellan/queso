from typing import Sequence

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

        self.state = qml.QNode(self._state, device=self.device, interface=self.interface)
        self.probs = qml.QNode(self._probs, device=self.device, interface=self.interface)
        self.sample = qml.QNode(self._sample, device=self.device, interface=self.interface)

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

    # def sensor(self, theta, phi, mu):
    #
    #     # preparation
    #     for j in range(self.k):
    #         for i in range(self.n):
    #             qml.RX(theta[i, 3 * j], wires=i)
    #             qml.RY(theta[i, 3 * j + 1], wires=i)
    #             qml.RZ(theta[i, 3 * j + 2], wires=i)
    #         for i in range(0, self.n - 1, 2):
    #             qml.CZ(wires=[i, i + 1])
    #         for i in range(1, self.n - 1, 2):
    #             qml.CZ(wires=[i, i + 1])
    #
    #     # interaction
    #     for i in range(self.n):
    #         qml.RX(phi, wires=i)
    #
    #     # detection
    #     for i in range(self.n):
    #         qml.RZ(mu[i, 0], wires=i)
    #         qml.RX(mu[i, 1], wires=i)
    #         qml.RY(mu[i, 2], wires=i)
    #     return qml

    def _state(self, theta, phi, mu):
        return self.sensor(theta, phi, mu).state()

    def _probs(self, theta, phi, mu):
        return self.sensor(theta, phi, mu).probs()

    def _sample(self, theta, phi, mu):
        return self.sensor(theta, phi, mu).sample()

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


n = 6
k = 4
key = jax.random.PRNGKey(0)
phi = jnp.array(0.1)
theta = jax.random.uniform(key, shape=[n, 3*k])
mu = jax.random.uniform(key, shape=[n, 3])
sensor = Sensor(n=n, k=k, shots=10)

state = sensor.state(theta, phi, mu)
print(state)

probs = sensor.probs(theta, phi, mu)
print(probs)
#
# samples = sensor.sample(theta, phi, mu, shots=10)
# print(samples)
#
# # for phi in jnp.linspace(0, jnp.pi, 32):
# #     samples = sensor.sample(theta, phi, mu, shots=10)
# #     print(samples)
#
# probs = sensor.probs(theta, phi, mu)
# print(probs)
#
fi = sensor.qfi(theta, phi, mu)
print(fi)

fi = sensor.cfi(theta, phi, mu)
print(fi)


#%%
def loss_qfi(params):
    return -sensor.qfi(params['theta'], phi, mu)


def loss_cfi(params):
    return -sensor.cfi(params['theta'], phi, params['mu'])


loss_fi = loss_qfi
params = {
    'theta': theta,
    # 'phi': phi,
    # 'mu': mu,
}

loss_val_grad = jax.value_and_grad(loss_fi)


#%%
@jax.jit
def step(params, opt_state):
    val, grads = loss_val_grad(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return val, params, updates, opt_state


#%%
lr = 1e-1
progress = True
optimizer = optax.adam(learning_rate=lr)
opt_state = optimizer.init(params)

val, grads = loss_val_grad(params)
print(sensor.qfi(theta, phi, mu))
print(val)

#%%
losses = []
for _ in range(1000):
    val, params, updates, opt_state = step(params, opt_state)
    # print(val)
    losses.append(-val)


losses = jnp.array(losses)

#%%
fig, ax = plt.subplots()
ax.plot(losses)
ax.axhline(n**2, ls='--', alpha=0.5)
plt.show()

# n_features = 2
# estimator = RegressionEstimator((n_features, 10, 10, 1))
# key = jax.random.PRNGKey(0)
# n_batch = 10
# batch = jax.random.uniform(key, shape=(n_batch, n_features))
# params = estimator.init(jax.random.PRNGKey(0), batch)
#
# y = estimator.apply(params, batch)
# print(y)
#
