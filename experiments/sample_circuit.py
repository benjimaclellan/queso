import time
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import jax
import jax.numpy as jnp
import optax

from queso.sensors.tc.sensor import Sensor, sample_bin2int
from queso.io import IO
import h5py


# %%
def sample_circuit(
        io: IO,
        n: int,
        k: int,
        key: jax.random.PRNGKey,
        phi_range: tuple,
        n_phis: int = 10,
        n_shots: int = 1000,
        n_shots_test: int = 500,
        plot: bool = False,
        progress: bool = True,
        **kwargs,
):
    # %%
    print(f"Initializing sensor n={n}, k={k}")
    sensor = Sensor(n, k, backend='ket', **kwargs)

    #%%
    hf = h5py.File(io.path.joinpath("circ.h5"), "r")
    # print(hf.keys())
    theta = jnp.array(hf.get("theta"))
    mu = jnp.array(hf.get("mu"))
    hf.close()

    # %%
    print(f"Sampling {n_shots} shots for {n_phis} phase value between {phi_range[0]} and {phi_range[1]}.")
    phis = (phi_range[1] - phi_range[0]) * jnp.arange(n_phis) / (n_phis - 1) + phi_range[0]

    t0 = time.time()
    shots, probs = sensor.sample_over_phases(theta, phis, mu, n_shots=n_shots + n_shots_test, verbose=True)
    t1 = time.time()
    print(f"Sampling took {t1 - t0} seconds.")

    shots_test = shots[:, -n_shots_test:]
    shots = shots[:, :n_shots]

    # %%
    outcomes = sample_bin2int(shots, n)
    counts = jnp.stack([jnp.count_nonzero(outcomes == x, axis=(1,), keepdims=True).squeeze() for x in range(2 ** n)], axis=1)
    freqs = counts / counts.sum(axis=-1, keepdims=True)

    # %%
    if plot:
        #%%
        fig, axs = plt.subplots(nrows=2)
        sns.heatmap(probs, ax=axs[0], cbar_kws={'label': 'True Probs.'})
        sns.heatmap(freqs, ax=axs[1], cbar_kws={'label': 'Rel. Freqs.'})
        plt.show()
        io.save_figure(fig, filename="probs_freqs.png")

    # %%
    hf = h5py.File(io.path.joinpath("samples.h5"), "w")
    hf.create_dataset("probs", data=probs)
    hf.create_dataset("shots", data=shots)
    hf.create_dataset("counts", data=counts)
    hf.create_dataset("shots_test", data=shots_test)
    hf.create_dataset("phis", data=phis)
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
