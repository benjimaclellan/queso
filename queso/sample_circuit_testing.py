import time
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

import jax
import jax.numpy as jnp
import optax

from queso.sensors.tc.sensor import Sensor, sample_bin2int
from queso.io import IO
from queso.configs import Configuration


# %%
def sample_circuit_testing(
        io: IO,
        config: Configuration,
        key: jax.random.PRNGKey,
        plot: bool = False,
        progress: bool = True,
):
    n = config.n
    k = config.k
    phis_test = jnp.array(config.phis_test)
    n_shots_test = config.n_shots_test
    kwargs = dict(preparation=config.preparation, interaction=config.interaction, detection=config.detection, backend=config.backend)

    # %%
    print(f"Initializing sensor n={n}, k={k}")
    sensor = Sensor(n, k, **kwargs)

    #%%
    hf = h5py.File(io.path.joinpath("circ.h5"), "r")
    theta = jnp.array(hf.get("theta"))
    mu = jnp.array(hf.get("mu"))
    hf.close()
    
    # %% testing samples
    print(f"Sampling {n_shots_test} shots for {phis_test}.")
    t0 = time.time()
    shots_test, probs_test = sensor.sample_over_phases(theta, phis_test, mu, n_shots=n_shots_test, verbose=True, key=key)
    t1 = time.time()
    print(f"Sampling took {t1 - t0} seconds.")

    # %%
    hf = h5py.File(io.path.joinpath("test_samples.h5"), "w")
    hf.create_dataset("probs_test", data=probs_test)
    hf.create_dataset("shots_test", data=shots_test)
    hf.create_dataset("phis_test", data=phis_test)
    hf.close()

    #%%
    return


if __name__ == "__main__":
    n = 1
    k = 1

    io = IO(folder=f"test-n{n}-k{k}", include_date=False)
    io.path.mkdir(parents=True, exist_ok=True)

    # %%
    key = jax.random.PRNGKey(time.time_ns())
    plot = True
