import jax
import jax.numpy as np
import tqdm
import matplotlib.pyplot as plt
import optax
from functools import partial
import uuid

from queso.sensors import Sensor
from queso.configs import Configuration


#%%
jax.config.update("jax_default_device", jax.devices("cpu")[0])
# jax.config.update("jax_default_device", jax.devices("gpu")[0])

config = Configuration(
    train_circuit=True
)

sensor = Sensor(
    n=4,
    k=4,
    kwargs=dict(
        preparation='hardware_efficient_ansatz'
    )
)

lr = 0.1
n_steps = 3000
progress = True

key = jax.random.PRNGKey(0)
theta = jax.random.uniform(key, sensor.theta.shape)
phi = jax.random.uniform(key, sensor.phi.shape)
mu = jax.random.uniform(key, sensor.mu.shape)
print(sensor.qfi(theta, phi))

qfi = jax.jit(partial(sensor.qfi, phi=phi))
print(qfi(theta))
print(qfi(theta).devices())

#%%
optimizer = optax.adagrad(learning_rate=lr)

params = {'theta': theta}
compute_loss_and_grads = jax.jit(jax.value_and_grad(lambda params: -sensor.qfi(theta=params['theta'], phi=phi)))
loss, grads = compute_loss_and_grads(params)
print(loss, grads)
opt_state = optimizer.init(params)

# #%%
# losses = []
# for step in (pbar := tqdm.tqdm(range(n_steps), disable=(not progress))):
#     loss, grads = compute_loss_and_grads(params)
#     updates, opt_state = optimizer.update(grads, opt_state)
#     params = optax.apply_updates(params, updates)
#     losses.append(-loss)
#     if progress:
#         pbar.set_description(f"Cost: {-loss:.10f}")
#
# #%%
# losses = np.array(losses)
#
# fig, axs = plt.subplots(1, 1)
# axs.axhline(sensor.n**2, **dict(color="teal", ls="--"))
# axs.plot(losses, **dict(color="salmon", ls="-"))
# axs.set(
#     xlabel="Optimization step",
#     ylabel=r"Fisher Information: $\mathcal{F}_\phi$",
# )
# plt.show()
