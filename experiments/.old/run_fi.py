import time
import tqdm
import matplotlib.pyplot as plt

import pennylane as qml
import jax
import jax.numpy as jnp
import optax


def probe(theta):
    for j in range(k):
        for i in range(n):
            qml.RX(theta[i, 3 * j], wires=i)
            qml.RY(theta[i, 3 * j + 1], wires=i)
            qml.RZ(theta[i, 3 * j + 2], wires=i)
        for i in range(0, n - 1, 2):
            qml.CNOT(wires=[i, i + 1])
        for i in range(1, n - 1, 2):
            qml.CNOT(wires=[i, i + 1])

    return qml


def interaction(phi):
    # interaction
    for i in range(n):
        qml.RZ(phi, wires=i)
    return qml


def detection(mu):
    for i in range(n):
        qml.RX(mu[i, 0], wires=i)
        qml.RY(mu[i, 1], wires=i)
        qml.RZ(mu[i, 2], wires=i)
    return qml


def sensor(theta, phi):
    for j in range(k):
        for i in range(n):
            qml.RX(theta[i, 3 * j], wires=i)
            qml.RY(theta[i, 3 * j + 1], wires=i)
            qml.RZ(theta[i, 3 * j + 2], wires=i)
        for i in range(0, n - 1, 2):
            qml.CZ(wires=[i, i + 1])
        for i in range(1, n - 1, 2):
            qml.CZ(wires=[i, i + 1])

    for i in range(n):
        qml.RZ(phi, wires=i)


def state(theta, phi):
    sensor(theta, phi)
    return qml.state()


n = 4
k = 6
interface = "jax"
device = qml.device("default.qubit", wires=n)

key = jax.random.PRNGKey(time.time_ns())
theta = jax.random.uniform(key, shape=[n, 3 * k])
phi = jnp.array(0.0)
mu = jax.random.uniform(key, shape=[n, 3])

qnode = qml.QNode(state, device=device, interface=interface)

# %%
psi = qnode(theta, phi)
dpsi = jax.jacrev(qnode, argnums=1, holomorphic=True)(
    theta.astype("complex64"), phi.astype("complex64")
)
print(psi)
print(jnp.conj(psi[None, :]) @ psi[:, None])


# %%
def qfi(theta, phi):
    psi = qnode(theta, phi)
    dpsi = jax.jacrev(qnode, argnums=1, holomorphic=True)(
        theta.astype("complex64"), phi.astype("complex64")
    )
    fi = (
        4
        * (
            jnp.conj(dpsi[None, :]) @ dpsi[:, None]
            - jnp.abs(jnp.conj(dpsi[None, :]) @ psi[:, None])
        ).squeeze()
    )
    return fi


def cost(params):
    fi = qfi(params["theta"], phi)
    return -jnp.real(fi)


cost_val_grad = jax.value_and_grad(cost)


# %%
@jax.jit
def step(params, opt_state):
    val, grads = cost_val_grad(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return val, params, updates, opt_state


# %%
params = {
    "theta": theta,
}

lr = 1e-1
progress = True
optimizer = optax.adam(learning_rate=lr)
opt_state = optimizer.init(params)

val, grads = cost_val_grad(params)
print(qfi(theta, phi))
print(val)


# %%
losses = []
for _ in range(1000):
    val, params, updates, opt_state = step(params, opt_state)
    print(val)
    losses.append(-val)


losses = jnp.array(losses)

# %%
fig, ax = plt.subplots()
ax.plot(losses)
plt.show()

# %%
