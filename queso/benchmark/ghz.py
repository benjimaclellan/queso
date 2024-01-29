import os
import time
import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from math import pi
import seaborn as sns
from queso.io import IO
from queso.train.vqs import vqs
from queso.configs import Configuration
from queso.sensors.tc.sensor import Sensor
from queso.benchmark.estimator import select_sample_sequence, posterior_product, bias, variance, estimate

#%%
def ghz_protocol(
    io: IO,
    config: Configuration,
):
    """
    Executes the GHZ protocol for a quantum sensor.

    This function sets up a Sensor object with the given configuration and performs a series of operations
    including sampling over phases, computing the GHZ estimator, and calculating biases and variances.
    It also generates plots for the posterior probabilities and the bias and variance as a function of sequence length.

    Args:
        io (IO): An instance of the IO class for handling input/output operations.
        config (Configuration): An instance of the Configuration class containing the settings for the GHZ protocol.

    Returns:
        None
    """
    #%%
    n = config.n
    seed = config.seed

    assert config.backend == "dm"
    assert config.preparation == "ghz_dephasing"
    assert config.detection == "hadamard_bases"

    #%%
    sensor = Sensor(
        n=n,
        k=None,
        **dict(
            preparation=config.preparation,
            interaction=config.interaction,
            detection=config.detection,
            gamma_dephasing=config.gamma_dephasing,
            backend=config.backend,
        )
    )

    #%%
    theta = sensor.theta
    phi = jnp.array(config.phi_fi)
    mu = sensor.mu

    #%% metrics
    metrics = {metric: None for metric in config.metrics}
    for metric in metrics.keys():
        if metric == "entropy_vn":
            metrics[metric] = sensor.entanglement(theta, phi)
        elif metric == "qfi":
            metrics[metric] = sensor.qfi(theta, phi)
        elif metric == "ghz_fidelity":
            state = sensor.state(theta, phi)
            if len(state.shape) == 1:  # ket
                fid = 0.5 * jnp.abs(state[0] + state[-1]) ** 2
            elif len(state.shape) == 2:  # density matrix
                fid = 0.5 * (state[0, 0] + state[-1, 0] + state[0, -1] + state[-1, -1])
            else:
                raise RuntimeError("State should always have 1 or 2 dims.")
            metrics[metric] = fid

    hf = io.save_h5("circ.h5")
    hf.create_dataset("fi_train", data=jnp.array(sensor.cfi(theta, phi, mu)))
    for metric, arr in metrics.items():
        hf.create_dataset(metric, data=arr)
    hf.close()

    # probs = sensor.probs(theta, phi, mu)
    # print(probs)
    # plt.figure()
    # plt.bar([i for i in range(len(probs))], probs)
    # plt.show()
    # print(f"gamma={gamma} | qfi = {sensor.qfi(theta, phi):0.5f} | cfi = {sensor.cfi(theta, phi, mu):0.5f}")
    # dpr = jax.jacrev(sensor.probs, argnums=1, holomorphic=False)(theta, phi, mu)
    # print(dpr)
    # shots = sensor.sample(theta, phi, mu, n_shots=100)
    # print(shots)
    # print(shots.sum(axis=-1) % 2)
    # phis = jnp.linspace(0.0, jnp.pi/2/n, 100)

    #%%
    def ghz_estimator(sequences, phis, n: int):
        """
        Maximum likelihood estimator for GHZ protocol.

        Args:
            sequences: shape of input array is [n_trials, n_phis_true, n_sequence_max, n_qubits]
            phis: array of phis to compute single-shot posterior pdf over

        Returns:

        """

        parity = sequences.sum(axis=-1) % 2  # compute the parity of the bitstring
        pred = (
            parity[:, :, :, None] * jnp.sin(n / 2 * phis[None, None, None, :])**2
            + jnp.logical_not(parity[:, :, :, None]) * jnp.cos(n / 2 * phis[None, None, None, :])**2
        )
        return pred

    #%%
    key = jax.random.PRNGKey(seed)

    n_trials = config.n_trials
    n_sequences = config.n_sequences
    phis_true = jnp.array(config.phis_test)
    n_shots_test = config.n_shots_test
    n_sequence_max = jnp.max(jnp.array(n_sequences))
    n_sequences = jnp.array(config.n_sequences)
    n_grid = config.n_grid

    #%%
    if config.sample_circuit_testing_data:
        print(f"Sampling {n_shots_test} shots for {phis_true}.")
        t0 = time.time()
        shots, probs = sensor.sample_over_phases(
            theta, phis_true, mu, n_shots=n_shots_test, verbose=True, key=key
        )
        t1 = time.time()
        print(f"Sampling took {t1 - t0} seconds.")

        hf = io.save_h5("test_samples.h5")
        hf.create_dataset("probs_test", data=probs)
        hf.create_dataset("shots_test", data=shots)
        hf.create_dataset("phis_test", data=phis_true)
        hf.close()

    #%%
    if config.benchmark_estimator:
        n_trials = config.n_trials
        n_sequences = jnp.array(config.n_sequences)
        n_grid = config.n_grid

        # %%
        hf = h5py.File(io.path.joinpath("test_samples.h5"), "r")
        shots = jnp.array(hf.get("shots_test"))
        phis_true = jnp.array(hf.get("phis_test"))
        hf.close()

        hf = h5py.File(io.path.joinpath("circ.h5"), "r")
        fi = jnp.array(hf.get("fi_train"))
        hf.close()

        keys = jax.random.split(key, n_trials)
        sequences = jnp.stack([select_sample_sequence(shots, key, n_sequence_max) for key in keys], axis=0)
        assert sequences.shape == (n_trials, len(phis_true), n_sequence_max, n)

        #%%
        # config.phi_center = 0.0  #jnp.pi/2/n
        # phis = jnp.linspace(-jnp.pi / 2 / n + config.phi_center, jnp.pi / 2 / n + config.phi_center, 100)
        phis = jnp.linspace(config.phi_range[0], config.phi_range[1], config.n_grid)
        pred = ghz_estimator(sequences, phis, n)

        #%%
        posteriors = jnp.stack(
            [posterior_product(pred, n_sequence) for n_sequence in n_sequences], axis=2
        )
        phis_estimates = estimate(posteriors, phis)
        biases = bias(phis_estimates, phis_true)
        variances = variance(posteriors, phis_estimates, phis)

        #%%
        for k in range(phis_true.shape[0]):
            print(phis_estimates[:, k, -1], phis_true[k])

        #%%
        k = 7
        fig, ax = plt.subplots()
        print(phis_true[k])
        for j in range(10):
            for i, n_sequence in enumerate(n_sequences):
                ax.plot(phis, posteriors[j, k, i, :])
        ax.axvline(phis_true[k])
        ax.set(xlabel=r"$\phi$")
        fig.show()

        colors = sns.color_palette('crest', n_colors=10)
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex='col')
        ax = axs[0]
        ax.errorbar(
            x=n_sequences,
            y=biases[:, k, :].mean(axis=0),
            xerr=None,
            yerr=jnp.var(biases[:, k, :], axis=0),
            color=colors[0],
            ls='-',
            lw=2,
            marker='o',
            elinewidth=2,
            ecolor=colors[2],
        )
        ax.axhline(0, color='black', ls='--')
        ax.set(xscale="log")

        ax = axs[1]
        ax.plot(
            n_sequences,
            variances[:, k, :].mean(axis=0),
            color=colors[0],
            ls='-',
            marker='o',
        )
        ax.plot(n_sequences, 1/(n_sequences * n), label='SQL, $(m n)^{-1}$', **dict(color='black', alpha=0.8, ls=':'))
        ax.plot(n_sequences, 1/(n_sequences * n**2), label='HL, $(m n^2)^{-1}$', **dict(color='black', alpha=0.8, ls='--'))
        ax.set(xscale="log", yscale='log')

        axs[0].legend(loc='lower right', bbox_to_anchor=(1.0,  0.0))
        axs[1].legend(loc='upper right', bbox_to_anchor=(1.0,  1.0))

        axs[0].set(ylabel="Bias,\n"+r"$\langle \bar{\phi} - \phi \rangle$")
        axs[1].set(ylabel="Variance,\n"+r"$\langle \Delta^2 \bar{\phi} \rangle$")
        axs[-1].set(xlabel="Sequence length, $m$")

        fig.show()


if __name__ == "__main__":
    n = 4
    io = IO(path=os.getenv("DATA_PATH"), folder="ghz_test")
    config = Configuration(
        n=n,
        preparation="ghz_dephasing",
        interaction="local_rz",
        detection="hadamard_bases",
        seed=123,
        gamma_dephasing=0.0,
        backend="ket",
        phi_fi=0.1,
    )
    config.phi_center = jnp.pi / 2 / n
    config.phis_test = jnp.linspace(-pi / 3 / n + config.phi_center, pi / 3 / n + config.phi_center, 9).tolist()

    ghz_protocol(io, config)