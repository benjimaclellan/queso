import time
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import jax
import jax.numpy as jnp
import optax

from queso.sensors.tc.sensor import Sensor
from queso.io import IO
import h5py


def train_circuit(
    io: IO,
    n: int,
    k: int,
    key: jax.random.PRNGKey,
    phi_range: tuple,
    n_phis: int = 10,
    lr: float = 1e-2,
    n_steps: int = 100,
    contractor: str = "auto",
    plot: bool = False,
    progress: bool = True,
):
    #%%
    print(f"Initializing sensor n={n}, k={k}")
    sensor = Sensor(n, k, backend='ket')
    phi = jnp.array(0.0)

    optimizer = optax.adam(learning_rate=lr)

    def loss_cfi(params, phi):
        return -sensor.cfi(params["theta"], phi, params["mu"])

    def loss_qfi(params, phi):
        return -sensor.qfi(params["theta"], phi)

    # %%
    @jax.jit
    def step(params, opt_state):
        val, grads = loss_val_grad(params, phi)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return val, params, updates, opt_state

    # %%
    theta = jax.random.uniform(key, shape=sensor.theta.shape)
    mu = jax.random.uniform(key, shape=sensor.mu.shape)

    loss = loss_cfi
    params = {"theta": theta, "mu": mu}

    # loss = loss_qfi
    # params = {'theta': theta}

    loss_val_grad = jax.value_and_grad(loss, argnums=0)

    opt_state = optimizer.init(params)
    # val, grads = loss_val_grad(params, phi)

    # %%
    losses, vn_ent_train = [], []
    for i in (pbar := tqdm.tqdm(range(n_steps), disable=(not progress))):
        val, params, updates, opt_state = step(params, opt_state)
        losses.append(-val)
        vn_ent_train.append(sensor.entanglement(params["theta"], phi))
        if progress:
            pbar.set_description(f"Step {i} | FI: {-val:.10f}")

    losses = jnp.array(losses)
    fi_train = losses  # set FI to the losses
    theta = params["theta"]
    mu = params["mu"]

    #%%
    if plot:
        # %% visualize
        fig, axs = plt.subplots(ncols=1, nrows=2, sharex=True)
        axs[0].plot(losses)
        axs[0].axhline(n**2, ls="--", alpha=0.5)
        axs[0].set(ylabel="Fisher Information")
        axs[1].plot(vn_ent_train)
        axs[1].set(ylabel="Entropy of entanglement", xlabel="Optimization Step")
        io.save_figure(fig, filename="fi-entropy-optimization")
        fig.show()

        #%%
        # sensor.circuit(theta, phi, mu).draw(output="text")

    #%% compute other quantities of interest and save
    phis = (phi_range[1] - phi_range[0]) * jnp.arange(n_phis) / (n_phis - 1) + phi_range[0]
    qfi_phis = jax.vmap(lambda phi: -loss_qfi(params={'theta': theta}, phi=phi))(phis)
    cfi_phis = jax.vmap(lambda phi: -loss_cfi(params={'theta': theta, "mu": mu}, phi=phi))(phis)

    # %%
    metadata = dict(n=n, k=k, lr=lr)
    io.save_json(metadata, filename="circ-metadata.json")

    hf = h5py.File(io.path.joinpath("circ.h5"), "w")
    hf.create_dataset("theta", data=theta)
    hf.create_dataset("mu", data=mu)
    hf.create_dataset("fi_train", data=fi_train)
    hf.create_dataset("vn_ent_train", data=vn_ent_train)
    hf.create_dataset("phis", data=phis)
    hf.create_dataset("qfi_phis", data=qfi_phis)
    hf.create_dataset("cfi_phis", data=cfi_phis)
    hf.close()

    return


#%%
if __name__ == "__main__":
    n = 1
    k = 1

    io = IO(folder=f"nn-estimator-n{n}-k{k}", include_date=True)
    io.path.mkdir(parents=True, exist_ok=True)

    # %%
    key = jax.random.PRNGKey(time.time_ns())
    plot = True
    train_circuit(
        io=io,
        n=n,
        k=k,
        key=key,
        n_steps=100,
        lr=1e-1,
        n_phis=200,
        plot=plot,
    )
    time.sleep(0.1)