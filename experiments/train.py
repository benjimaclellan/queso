from typing import Sequence
import itertools
import time
import tqdm
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn

from queso.tc.sensor import Sensor


n = 4
k = 6
sensor = Sensor(n, k)

key = jax.random.PRNGKey(1234)

phi = jnp.array(0.0)
theta = jax.random.uniform(key, shape=[n, k, 2])
mu = jax.random.uniform(key, shape=[n, 3])

# t0 = time.time()
fi = sensor.cfi(theta, phi, mu)
# t1 = time.time()
fi = sensor.qfi(theta, phi)
# t2 = time.time()
# print(f"{t1-t0} | {t2-t1}")

lr = 1e-1
progress = True
optimizer = optax.adam(learning_rate=lr)


def loss_cfi(params):
    return -sensor.cfi(params['theta'], phi, params['mu'])


def loss_qfi(params):
    return -sensor.qfi(params['theta'], phi)


def loss_entanglement(params):
    return -sensor.entanglement(params['theta'], phi)


#%%
@jax.jit
def step(params, opt_state):
    val, grads = loss_val_grad(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return val, params, updates, opt_state


#%%
# key = jax.random.PRNGKey(1234)
key = jax.random.PRNGKey(time.time_ns())

phi = jnp.array(0.0)
theta = jax.random.uniform(key, shape=[n, k, 2])
mu = jax.random.uniform(key, shape=[n, 3])


loss = loss_cfi
params = {'theta': theta, 'mu': mu}

# loss = loss_qfi
# loss = loss_entanglement
# params = {'theta': theta}

loss_val_grad = jax.value_and_grad(loss)

opt_state = optimizer.init(params)
val, grads = loss_val_grad(params)


#%%
losses, entropies = [], []
for _ in range(100):
    val, params, updates, opt_state = step(params, opt_state)
    print(val)
    losses.append(-val)
    entropies.append(jnp.exp(sensor.entanglement(params['theta'], phi)))

losses = jnp.array(losses)

## %%
fig, axs = plt.subplots(ncols=1, nrows=2, sharex=True)
axs[0].plot(losses)
axs[0].axhline(n**2, ls='--', alpha=0.5)
axs[0].set(ylabel="Fisher Information")
axs[1].plot(entropies)
axs[1].set(ylabel="Entropy of entanglement", xlabel="Optimization Step")
plt.show()

sensor.circuit(theta, phi, mu).draw(output="text")

# #%%
# model = RegressionEstimator((n, 10, 10, 1))
# key = jax.random.PRNGKey(0)
# batch_size = 100
# x = jnp.array(list(counts.values()))
# y = jnp.array(list(counts.keys()))
#
#
# # inds = jax.random.permutation(key, jnp.array(range(y.shape[0])), axis=0, independent=True)
# # batch_size = 10
# # batches = [inds[i:i+batch_size] for i in range(0, len(inds), batch_size)]
# params = model.init(jax.random.PRNGKey(0), x)
# # batch = jax.random.shuffle(, shape=(n_batch, n))
# # batch = x[inds[0:10], :], y[inds[0:10]]
# pred = model.apply(params, x)
# print(pred)
#
#
# #%%
# def mse(params, x, y):
#     # Define the squared loss for a single pair (x,y)
#     def squared_error(x, y):
#         pred = model.apply(params, x)
#         return jnp.inner(y-pred, y-pred) / 2.0
# # Vectorize the previous to compute the average of the loss on all samples.
#     return jnp.mean(jax.vmap(squared_error)(x, y), axis=0)
#
#
# #%%
# # print(mse(params, batch[0], batch[1]))
#
# #%%
# loss_val_grad = jax.value_and_grad(mse)
# # print(loss_val_grad(params, x, y))
#
#
# #%%
# @jax.jit
# def step_nn(params, x, y, opt_state):
#     val, grads = loss_val_grad(params, x, y)
#     updates, opt_state = optimizer.update(grads, opt_state)
#     params = optax.apply_updates(params, updates)
#     return val, params, updates, opt_state
#
#
# #%%
# lr = 0.5e-1
# progress = True
# # batch_size = 10
#
# optimizer = optax.adam(learning_rate=lr)
# opt_state = optimizer.init(params)
#
#
# #%%
# # def batch_generator(x, y, batch_size):
# #     inds = jax.random.permutation(key, jnp.array(range(y.shape[0])), axis=0, independent=True)
# #     batches = [inds[i:i + batch_size] for i in range(0, len(inds), batch_size)]
# #     return [(x[batch, :], y[batch]) for batch in batches]
# #
# #
# # batches = batch_generator(x, y, batch_size)
#
#
# #%%
# losses = []
# for epoch in range(1000):
#     # batches = batch_generator(x, y, batch_size)
#     # for i, batch in enumerate(batch_generator(x, y, batch_size=batch_size)):
#     # val, params, updates, opt_state = step_nn(params, batch[0], batch[1], opt_state)
#     val, params, updates, opt_state = step_nn(params, x, y, opt_state)
#     print(f"Epoch {epoch} | MSE: {val} | {x.shape}")
#     losses.append(val)
#
# losses = jnp.array(losses)
#
# #%%
#
# fig, ax = plt.subplots()
# ax.plot(losses)
# plt.show()
#
# #%%
# pred = model.apply(params, x)
# fig, ax = plt.subplots()
# ax.scatter(y, pred)
# plt.show()

#%%