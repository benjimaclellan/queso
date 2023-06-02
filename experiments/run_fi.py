import jax
import jax.numpy as jnp
import pennylane as qml
import optax
import tqdm


def probe(theta):
    for j in range(k):
        for i in range(n):
            qml.RX(theta[i, 3*j], wires=i)
            qml.RY(theta[i, 3*j + 1], wires=i)
            qml.RZ(theta[i, 3*j + 2], wires=i)
        for i in range(0, n-1, 2):
            qml.CNOT(wires=[i, i+1])
        for i in range(1, n-1, 2):
            qml.CNOT(wires=[i, i+1])

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


def qsense(theta, phi):
    probe(theta)
    interaction(phi)
    return qml.state()


def csense(theta, phi, mu):
    probe(theta)
    interaction(phi)
    detection(mu)
    return qml.probs()


n = 2
k = 2
interface = "jax"
device = qml.device("default.qubit", wires=n)

key = jax.random.PRNGKey(0)

theta = jax.random.uniform(key, shape=[n, 3*k]).astype("complex64")
# phi = jax.random.uniform(key).astype("complex64")
mu = jax.random.uniform(key, shape=[n, 3]).astype("complex64")

# theta = jnp.zeros([n, 3*k]).astype("complex64")
phi = complex(1)  #jnp.zeros([1]).astype("complex64")
# mu = jnp.zeros([n, 3]).astype("complex64")

sensor = qml.QNode(qsense, device=device, interface=interface)
state_jit = jax.jit(sensor)
state_grad_jit = jax.jit(jax.jacrev(sensor, argnums=1, holomorphic=True))

#%%
# psi = state_jit(theta, phi)
# dpsi = state_grad_jit(theta, phi)
# print(psi)
# print(jnp.conj(dpsi[None, :]) @ dpsi[:, None])
# print(dpsi)


#%%
def qfi(theta, phi):
    psi = state_jit(theta, phi)
    dpsi = state_grad_jit(theta, phi)
    fi = 4 * (jnp.conj(dpsi[None, :]) @ dpsi[:, None] - jnp.abs(jnp.conj(dpsi[None, :]) @ psi[:, None])).squeeze()
    return fi


theta = jax.random.uniform(key, shape=[n, 3*k]).astype("complex64")

print(qfi(theta, phi))



#%%
qfi_val_grad_jit = jax.jit(jax.value_and_grad(qfi, argnums=0, holomorphic=True))
fi, dfi = qfi_val_grad_jit(theta, phi)
print(dfi)

#%%
lr = 1e-2
progress = True
optimizer = optax.adagrad(learning_rate=lr)

params = {
    'theta': theta,
}

opt_state = optimizer.init(params)

#%%
def cost(params: optax.Params):
    fi = qfi(params["theta"], phi)
    # assert jnp.isclose(jnp.imag(fi), 0)
    return jnp.real(fi)

cost_val_grad = jax.jit(jax.value_and_grad(cost))
val, grads = cost_val_grad(params)
print(val)

#%%
n_steps = 1000
losses = []
for step in (pbar := tqdm.tqdm(range(n_steps), disable=(not progress))):
    val, grads = cost_val_grad(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    losses.append(val)
losses = jnp.array(losses)
print(losses)
