
from functools import partial

import quimb as qu
import quimb.tensor as qtn

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax


qtn.contract_backend("jax", set_globally=True)
qtn.set_contract_backend("jax")


class Sensor:
    def __init__(self, n, k):
        self.n = n
        self.k = k

    # def setup(self):
    #     # strip out the initial raw arrays
    #     params, skeleton = qtn.pack(psi)
    #     # save the stripped 'skeleton' tn for use later
    #     self.skeleton = skeleton
    #
    #     # assign each array as a parameter to optimize
    #     self.params = {
    #         i: self.param(f'param_{i}', lambda _: data)
    #         for i, data in params.items()
    #     }

    @partial(jax.jit, static_argnums=(0,))
    def state(self, theta, phi):
        # self.circ(theta, phi)
        return self.circuit(theta, phi).psi.to_dense()

    def circuit(self, theta, phi):
        circ = qtn.Circuit(self.n)
        for j in range(self.k):
            for i in range(self.n):
                circ.apply_gate('U3', *theta[i, j, :], i, gate_round=None, parametrize=True)
            for i in range(0, self.n - 1, 2):
                circ.apply_gate('CNOT', i, i + 1)
            for i in range(1, self.n - 1, 2):
                circ.apply_gate('CNOT', i, i + 1)

        for i in range(n):
            circ.apply_gate('RX', phi, i, gate_round=None, parametrize=True)

        return circ

    # @partial(jax.jit, static_argnums=(0,))
    def sample(self, shots, theta, phi):
        return self.circuit(theta, phi).sample(shots)
    # def __call__(self):
    #     psi = qtn.unpack(self.params, self.skeleton)
    #     return loss_fn(norm_fn(psi))


n = 4
k = 10

circ = qtn.Circuit(n)
key = jax.random.PRNGKey(2)
theta = jax.random.uniform(key, shape=(n, k, 3))
phi = jnp.array(0.0)
mu = jax.random.uniform(key, shape=(n, 3))


sensor = Sensor(n, k)


print(type(sensor.state(theta, phi)))

print(jnp.sum(jnp.abs(sensor.state(theta, phi))**2))
sensor.sample(1, theta, phi)
for b in sensor.circuit(theta, phi).sample(10):
    print(b)

# for j in range(k):
#     for i in range(n):
#         circ.apply_gate('U3', *theta[i, j, :], i, gate_round=None, parametrize=True)
#
#     for i in range(n - 1):
#         circ.apply_gate('CNOT', i, i+1)
#
# print(type(circ.psi.to_dense()))

# for b in circ.sample(80):
#     print(b)

# print(circ.to_dense())
# model = Sensor(n=2, k=2)
# params = model.init(jax.random.PRNGKey(42))
# loss_grad_fn = jax.value_and_grad(model.apply)
#
# # initialize our optimizer
# tx = optax.adabelief(learning_rate=0.01)
# opt_state = tx.init(params)
#
# @jax.jit
# def step(params, opt_state):
#     # our step: compute the loss and gradient, and update the optimizer
#     loss, grads = loss_grad_fn(params)
#     updates, opt_state = tx.update(grads, opt_state, params)
#     params = optax.apply_updates(params, updates)
#     return params, opt_state, loss
