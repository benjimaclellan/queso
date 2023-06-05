from typing import Sequence

import time
import tqdm
import matplotlib.pyplot as plt

import pennylane as qml
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn

from queso.sensor import Sensor


n = 6
k = 4
key = jax.random.PRNGKey(0)
phi = jnp.array(0.0)
theta = jax.random.uniform(key, shape=[n, 3*k])
mu = jax.random.uniform(key, shape=[n, 3])
sensor = Sensor(n=n, k=k, shots=10)

state = sensor.state(theta, phi, mu)
print(state)

probs = sensor.probs(theta, phi, mu)
print(probs)

samples = sensor.sample(theta, phi, mu, shots=10)
print(samples)
#

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


loss_fi = loss_cfi
params = {
    'theta': theta,
    # 'phi': phi,
    'mu': mu,
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

#%%
losses = []
for _ in range(1000):
    val, params, updates, opt_state = step(params, opt_state)
    losses.append(-val)

losses = jnp.array(losses)

#%%
fig, ax = plt.subplots()
ax.plot(losses)
ax.axhline(n**2, ls='--', alpha=0.5)
plt.show()

#%%
theta = params['theta']
mu = params['mu']
samples = sensor.sample(theta, phi, mu, shots=10)


def sweep_sample(sensor, theta, mu, n_points, n_samples):
    samples = {}
    for phi in jnp.linspace(0, jnp.pi, n_points):
        samples[float(phi)] = sensor.sample(theta, phi, mu, shots=n_samples)
    return samples


#%%
samples = sweep_sample(sensor, theta, mu, n_points=10, n_samples=100)

#%%
# n_features = 2
# estimator = RegressionEstimator((n_features, 10, 10, 1))
# key = jax.random.PRNGKey(0)
# n_batch = 10
# batch = jax.random.uniform(key, shape=(n_batch, n_features))
# params = estimator.init(jax.random.PRNGKey(0), batch)
#
# y = estimator.apply(params, batch)
# print(y)

#%%
