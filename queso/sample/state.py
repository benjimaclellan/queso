import time
import tqdm
import matplotlib.pyplot as plt
import tensorcircuit as tc
from functools import partial

import jax
import jax.numpy as jnp
import optax

from queso.sensors.tc.sensor import Sensor
from queso.io import IO
import h5py

backend = tc.set_backend("jax")
tc.set_dtype("complex128")
tc.set_contractor("auto")


def sample_state(
    io: IO,
    n: int,
    key: jax.random.PRNGKey,
    n_phis: int = 10,
    n_shots: int = 500,
):
    print(f"Initializing sensor n={n}, {state}")

    # %%
    if state == "ghz":

        def circuit(n, phi):
            c = tc.Circuit(n)
            c.h(0)
            for i in range(1, n):
                c.cnot(0, i)
            for i in range(n):
                c.rz(i, theta=phi)
            for i in range(n):
                c.h(i)
            return c

    else:
        raise ValueError

    @partial(jax.jit, static_argnums=(0,))
    def _sample(n, phi, key):
        backend.set_random_state(key)
        c = circuit(n, phi)
        return c.measure(*list(range(n)))[0]

    @partial(jax.jit, static_argnums=(0,))
    def probability(n, phi):
        c = circuit(n, phi)
        return c.probability()

    def sample(n, phi, key=None, n_shots=100):
        if key is None:
            key = jax.random.PRNGKey(time.time_ns())
        keys = jax.random.split(key, n_shots)
        shots = jnp.array([_sample(n, phi, key) for key in keys]).astype("int8")
        return shots

    def sample_over_phases(n, phis, n_shots, key=None):
        if key is None:
            key = jax.random.PRNGKey(time.time_ns())
        keys = jax.random.split(key, phis.shape[0])
        data = jnp.stack(
            [
                sample(n, phi, key=key, n_shots=n_shots)
                for (phi, key) in zip(phis, keys)
            ],
            axis=0,
        )
        probs = jnp.stack([probability(n, phi) for phi in phis], axis=0)
        return data, probs

    # %%
    print(f"Sampling {n_shots} shots for {n_phis} phase value between 0 and pi.")
    phis = jnp.linspace(0, jnp.pi, n_phis, endpoint=False)
    t0 = time.time()
    shots, probs = sample_over_phases(n, phis, n_shots=n_shots)
    t1 = time.time()
    print(f"Sampling took {t1 - t0} seconds.")

    # %%
    metadata = dict(n=n, state=state)
    io.save_json(metadata, filename="circ-metadata.json")

    hf = h5py.File(io.path.joinpath("circ.h5"), "w")
    hf.create_dataset("phis", data=phis)
    hf.create_dataset("shots", data=shots)
    hf.create_dataset("probs", data=probs)
    hf.close()

    return


if __name__ == "__main__":
    n = 1
    state = "ghz"

    io = IO(folder=f"calibration-samples-n{n}-{state}", include_date=True)
    io.path.mkdir(parents=True, exist_ok=True)

    # %%
    key = jax.random.PRNGKey(time.time_ns())
    sample_state(
        io=io,
        n=n,
        key=key,
        n_phis=200,
        n_shots=1000,
    )
    time.sleep(0.1)
