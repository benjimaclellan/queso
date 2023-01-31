import jax
import jax.numpy as np
import tqdm
import matplotlib.pyplot as plt
import optax
from functools import partial
import uuid

from queso.sensor.blocks import Probe, Interaction, Measurement, Sensor
from queso.sensor.unitaries import U3, Phase
from queso.quantities.information import neg_qfi, neg_cfi
from queso.sensor.layers import brick_wall_probe


def optimize_fi(n, fi, n_layers=4, n_runs=1, n_steps=300, lr=0.05, progress=True):
    probe = brick_wall_probe(n=n, d=2, n_layers=n_layers)

    interaction = Interaction(n=n)
    interaction.add([Phase("phi") for _ in range(n)])

    measurement = Measurement(n=n)
    measurement.add([U3(str(uuid.uuid4())) for _ in range(n)])

    sensor = Sensor(probe, interaction, measurement)
    params = sensor.initialize()

    fi = jax.jit(partial(fi, sensor=sensor, key="phi"))

    optimizer = optax.adam(learning_rate=lr)
    grad = jax.jit(jax.grad(fi))
    _ = grad(params)
    losses, _params = [], []
    for run in range(n_runs):
        loss = []

        params = sensor.initialize()
        opt_state = optimizer.init(params)

        for step in (pbar := tqdm.tqdm(range(n_steps), disable=(not progress))):
            ell = fi(params)
            gradient = grad(params)
            updates, opt_state = optimizer.update(gradient, opt_state)
            params = optax.apply_updates(params, updates)
            loss.append(ell)

            if progress:
                pbar.set_description(f"Cost: {-ell:.10f}")

        loss = -np.array(loss)

        losses.append(loss)
        _params.append(params)
    return losses, _params


if __name__ == "__main__":
    n = 6
    lr = 0.05
    n_steps = 500
    n_runs = 1
    n_layers = 8

    losses_cfi, _ = optimize_fi(
        n=n,
        fi=neg_cfi,
        n_layers=n_layers,
        n_runs=n_runs,
        n_steps=n_steps,
        lr=lr,
        progress=True,
    )
    losses_qfi, _ = optimize_fi(
        n=n,
        fi=neg_qfi,
        n_layers=n_layers,
        n_runs=n_runs,
        n_steps=n_steps,
        lr=lr,
        progress=True,
    )

    fig, axs = plt.subplots(1, 1)
    axs.axhline(n**2, **dict(color="gray", ls="--"))
    for loss in losses_cfi:
        axs.plot(loss, label="Classical FI", **dict(color="salmon", ls="-"))
    for loss in losses_qfi:
        axs.plot(loss, label="Quantum FI", **dict(color="teal", ls="-"))
    axs.set(
        xlabel="Optimization step",
        ylabel=r"Fischer Information: $\mathcal{F}_\phi$",
    )
    plt.show()
