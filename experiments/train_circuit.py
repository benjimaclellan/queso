import itertools
import time
import tqdm
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax

from queso.tc.sensor import Sensor
from queso.io import IO
import h5py


def train_circuit(io: IO, n: int, k: int, key: jax.random.PRNGKey):
    sensor = Sensor(n, k)

    phi = jnp.array(0.0)
    theta = jax.random.uniform(key, shape=[n, k, 2])
    mu = jax.random.uniform(key, shape=[n, 3])

    lr = 1e-2
    progress = True
    optimizer = optax.adam(learning_rate=lr)

    def loss_cfi(params):
        return -sensor.cfi(params['theta'], phi, params['mu'])


    def loss_qfi(params):
        return -sensor.qfi(params['theta'], phi)


    #%%
    @jax.jit
    def step(params, opt_state):
        val, grads = loss_val_grad(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return val, params, updates, opt_state


    #%%
    key = jax.random.PRNGKey(time.time_ns())

    phi = jnp.array(0.0)
    theta = jax.random.uniform(key, shape=[n, k, 2])
    mu = jax.random.uniform(key, shape=[n, 3])

    loss = loss_cfi
    params = {'theta': theta, 'mu': mu}

    # loss = loss_qfi
    # params = {'theta': theta}

    loss_val_grad = jax.value_and_grad(loss)

    opt_state = optimizer.init(params)
    val, grads = loss_val_grad(params)

    #%%
    losses, entropies = [], []
    for _ in range(1000):
        val, params, updates, opt_state = step(params, opt_state)
        print(val)
        losses.append(-val)
        entropies.append(sensor.entanglement(params['theta'], phi))

    losses = jnp.array(losses)

    theta = params['theta']
    mu = params['mu']

    # %%
    fig, axs = plt.subplots(ncols=1, nrows=2, sharex=True)
    axs[0].plot(losses)
    axs[0].axhline(n**2, ls='--', alpha=0.5)
    axs[0].set(ylabel="Fisher Information")
    axs[1].plot(entropies)
    axs[1].set(ylabel="Entropy of entanglement", xlabel="Optimization Step")
    io.save_figure(fig, filename="fi-entropy-optimization")

    # sensor.circuit(theta, phi, mu).draw(output="text")

    #%%
    n_phis = 10
    n_shots = 500
    phis = jnp.linspace(0, jnp.pi, n_phis)
    shots = sensor.sample_over_phases(theta, phis, mu, n_shots=n_shots)

    #%%
    hf = h5py.File(io.path.joinpath("test.h5"), 'w')
    hf.create_dataset('n', data=n)
    hf.create_dataset('k', data=k)
    hf.create_dataset('theta', data=theta)
    hf.create_dataset('mu', data=mu)
    hf.create_dataset('phis', data=phis)
    hf.create_dataset('shots', data=shots)
    hf.create_dataset('losses', data=losses)
    hf.close()

    return
