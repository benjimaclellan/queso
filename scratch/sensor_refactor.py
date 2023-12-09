import jax
import jax.numpy as np
import tqdm
import matplotlib.pyplot as plt
import optax
from functools import partial
import uuid

from queso.sensors.pennylane.sensor import Probe, Interaction, Measurement
from queso.sensors.pennylane.sensor import Sensor
from queso.sensors.pennylane.sensor import U3, CNOT, Identity, Phase
from queso.old.quantities import neg_cfi


if __name__ == "__main__":
    # io = IO(folder="qfi-optimization", include_date=True)

    n = 4  # number of particles
    d = 2
    n_layers = 4
    lr = 0.05
    n_steps = 300
    progress = True

    probe = Probe(n=n)
    for i in range(n_layers):
        probe.add([U3(str(uuid.uuid4())) for _ in range(n)])
        probe.add([CNOT(str(uuid.uuid4())) for _ in range(0, n, 2)])
        probe.add([Identity()] + [CNOT(str(uuid.uuid4())) for _ in range(1, n-1, 2)] + [Identity()])

    interaction = Interaction(n=n)
    interaction.add([Phase("phi") for _ in range(n)])

    measurement = Measurement(n=n)
    measurement.add([U3(str(uuid.uuid4())) for _ in range(n)])

    sensor = Sensor(probe, interaction, measurement)
    params = sensor.initialize()

    cfi = jax.jit(partial(neg_cfi, sensor=sensor, key="phi"))

    optimizer = optax.adagrad(learning_rate=lr)
    grad = jax.jit(jax.grad(cfi))
    _ = grad(params)

    _losses, _params = [], []
    for run in range(3):
        losses = []

        params = sensor.initialize()
        opt_state = optimizer.init(params)

        for step in (pbar := tqdm.tqdm(range(n_steps), disable=(not progress))):
            ell = cfi(params)
            gradient = grad(params)
            updates, opt_state = optimizer.update(gradient, opt_state)
            params = optax.apply_updates(params, updates)
            losses.append(ell)

            if progress:
                pbar.set_description(f"Cost: {-ell:.10f}")

        losses = -np.array(losses)

        _losses.append(losses)
        _params.append(params)

    fig, axs = plt.subplots(1, 1)
    axs.axhline(n**d, **dict(color="teal", ls="--"))
    for losses in _losses:
        axs.plot(losses, **dict(color="salmon", ls="-"))
    axs.set(
        xlabel="Optimization step",
        ylabel=r"Quantum Fischer Information: $\mathcal{F}_\phi$",
    )
    plt.show()
