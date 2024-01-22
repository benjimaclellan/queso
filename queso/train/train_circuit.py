import time
import os
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import argparse

import jax
import jax.numpy as jnp
import optax

from queso.sensors.tc.sensor import Sensor
from queso.io import IO
from queso.configs import Configuration


# %%
def train_circuit(
    io: IO,
    config: Configuration,
    key: jax.random.PRNGKey,
    plot: bool = False,
    progress: bool = True,
):
    """
    Trains a quantum circuit based on the provided configuration.

    This function initializes a sensor, sets up the optimizer and loss function, and then performs the training steps.
    It also saves the training results and metadata, and optionally plots the training progress.

    Args:
        io (IO): An instance of the IO class for handling input/output operations.
        config (Configuration): An instance of the Configuration class containing the settings for the training.
        key (jax.random.PRNGKey): A random number generator key from JAX.
        plot (bool, optional): If True, plots of the training progress are generated and saved. Defaults to False.
        progress (bool, optional): If True, a progress bar is displayed during training. Defaults to True.

    Returns:
        None

    Raises:
        ValueError: If the specified Fisher Information loss function is not recognized.
    """
    jax.config.update("jax_default_device", jax.devices(os.getenv("DEFAULT_DEVICE_TRAIN_CIRC", "cpu"))[0])

    # %%
    n = config.n
    k = config.k
    phi_range = config.phi_range
    n_phis = config.n_phis
    lr = config.lr_circ
    n_steps = config.n_steps
    kwargs = dict(
        preparation=config.preparation,
        interaction=config.interaction,
        detection=config.detection,
        backend=config.backend,
        n_ancilla=config.n_ancilla,
        gamma_dephasing=config.gamma_dephasing,
    )

    # %%
    print(f"Initializing sensor n={n}, k={k}")
    sensor = Sensor(n, k, **kwargs)
    phi = jnp.array(config.phi_center)
    theta = jax.random.uniform(key, shape=sensor.theta.shape)
    mu = jax.random.uniform(key, shape=sensor.mu.shape)

    # %%
    print(f"plot = {plot}")
    if plot:
        try:
            fig = sensor.circuit(theta, phi, mu).draw(**dict(output="mpl"))
            io.save_figure(fig, filename="circuit")
            fig.show()
        except:
            pass

    # %%
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

    if config.loss_fi == "loss_cfi":
        loss = loss_cfi
        params = {"theta": theta, "mu": mu}
    elif config.loss_fi == "loss_qfi":
        loss = loss_qfi
        params = {"theta": theta}
    else:
        raise ValueError(
            "Not a valid Fisher Information loss function. Use loss_cfi or loss_qfi."
        )

    loss_val_grad = jax.value_and_grad(loss, argnums=0)

    opt_state = optimizer.init(params)
    # val, grads = loss_val_grad(params, phi)

    # %%
    def metrics_callback(metrics: dict, params, phi):
        for metric in metrics.keys():
            if metric == "entropy_vn":
                metrics[metric].append(sensor.entanglement(params["theta"], phi))
            elif metric == "qfi":
                metrics[metric].append(sensor.qfi(params["theta"], phi))

    metrics = {metric: [] for metric in config.metrics}
    losses = []
    for i in (pbar := tqdm.tqdm(range(n_steps), disable=(not progress))):
        val, params, updates, opt_state = step(params, opt_state)
        losses.append(-val)
        metrics_callback(metrics, params, phi)
        if progress:
            pbar.set_description(f"Step {i} | FI: {-val:.10f}")

    losses = jnp.array(losses)
    fi_train = losses  # set FI to the losses
    theta = params["theta"]

    if config.loss_fi == "loss_cfi":
        mu = params["mu"]

    hf = io.save_h5("circ.h5")
    hf.create_dataset("theta", data=theta)
    hf.create_dataset("mu", data=mu)
    hf.create_dataset("fi_train", data=fi_train)
    for metric, arr in metrics.items():
        hf.create_dataset(metric, data=arr)
    hf.close()

    # %% compute other quantities of interest and save
    phis = (phi_range[1] - phi_range[0]) * jnp.arange(n_phis) / (
        n_phis - 1
    ) + phi_range[0]
    # qfi_phis = jax.vmap(lambda phi: -loss_qfi(params={'theta': theta}, phi=phi))(phis)  # causes memory issues sometimes
    # cfi_phis = jax.vmap(lambda phi: -loss_cfi(params={'theta': theta, "mu": mu}, phi=phi))(phis)

    # qfi_phis = jnp.array([-loss_qfi(params={'theta': theta}, phi=phi) for phi in phis])
    # cfi_phis = jnp.array([-loss_cfi(params={'theta': theta, "mu": mu}, phi=phi) for phi in phis])

    # %%
    metadata = dict(n=n, k=k, lr=lr)
    metadata.update(sensor.layers)
    io.save_json(metadata, filename="circ-metadata.json")

    hf = h5py.File(io.path.joinpath("circ.h5"), "a")
    hf.create_dataset("phis", data=phis)
    # hf.create_dataset("qfi_phis", data=qfi_phis)
    # hf.create_dataset("cfi_phis", data=cfi_phis)
    hf.close()

    try:
        qasm = sensor.circuit(theta, phi, mu).to_openqasm()
        io.save_txt(qasm, filename="circ.qasm")
    except:
        print("Could not save QASM text file.")

    if plot:
        # %% visualize
        fig, axs = plt.subplots(ncols=1, nrows=2, sharex=True)
        axs[0].plot(losses)
        axs[0].axhline(n**2, ls="--", alpha=0.5)
        axs[0].set(ylabel="Fisher Information")
        try:
            axs[0].plot(metrics['qfi'])
        finally:
            pass

        try:
            axs[1].plot(metrics['entropy_vn'])
        finally:
            pass
        axs[1].set(ylabel="Entropy of entanglement", xlabel="Optimization Step")
        io.save_figure(fig, filename="fi-entropy-optimization")
        fig.show()

        # #%%
        # fig, axs = plt.subplots(ncols=1, nrows=2, sharex=True)
        # # axs[0].plot(phis / jnp.pi, qfi_phis)
        # axs[1].plot(phis / jnp.pi, cfi_phis)
        # # axs[0].set(ylabel="QFI")
        # axs[1].set(ylabel="CFI")
        # axs[-1].set(xlabel=r"$\phi$ (rad/$\pi$)")
        # # for ax in axs:
        # #     ax.set(ylim=[0, 1.1 * n**2])
        # io.save_figure(fig, filename="qfi-cfi-phi")
        # fig.show()

    print(f"Finished training the circuits.")

    return


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="tmp")
    args = parser.parse_args()
    folder = args.folder

    io = IO(folder=f"{folder}")
    print(io)
    config = Configuration.from_yaml(io.path.joinpath("config.yaml"))
    key = jax.random.PRNGKey(config.seed)
    print(f"Training circuit: {folder} | Devices {jax.devices()} | Full path {io.path}")
    print(f"Config: {config}")
    train_circuit(io, config, key, progress=True, plot=True)
