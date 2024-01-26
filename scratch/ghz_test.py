import os
import time
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
if __name__ == "__main__":
    #%%
    # gammas = jnp.logspace(-5, -0.5, 10)
    n = 6
    io = IO(path=os.getenv("DATA_PATH"), folder="test_ghz")
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

    #%%
    # gammas = [0.000001]
    gamma = 0.4
    # for gamma in gammas:
    sensor = Sensor(
        n=n,
        k=4,
        **dict(
            preparation='ghz_dephasing',
            # interaction='local_rx',
            interaction='local_rz',
            # detection='computational_bases',
            detection='hadamard_bases',
            gamma_dephasing=gamma,
            backend='dm',
        )
    )

    #%%
    theta = sensor.theta
    # phi = jnp.array(jnp.pi/2/n)
    phi = jnp.array(0.1)
    mu = sensor.mu

    # probs = sensor.probs(theta, phi, mu)
    # print(probs)
    # plt.figure()
    # plt.bar([i for i in range(len(probs))], probs)
    # plt.show()
    # print(f"gamma={gamma} | qfi = {sensor.qfi(theta, phi):0.5f} | cfi = {sensor.cfi(theta, phi, mu):0.5f}")
    # dpr = jax.jacrev(sensor.probs, argnums=1, holomorphic=False)(theta, phi, mu)
    # print(dpr)

    # shots = sensor.sample(theta, phi, mu, n_shots=100)
    #
    # print(shots)
    # print(shots.sum(axis=-1) % 2)
    #
    # phis = jnp.linspace(0.0, jnp.pi/2/n, 100)

    #%%
    def ghz_estimator(sequences, phis, n: int):
        """

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
    key = jax.random.PRNGKey(0)

    n_trials = config.n_trials
    n_sequences = config.n_sequences
    phis_true = jnp.array(config.phis_test)
    n_shots_test = config.n_shots_test
    n_sequence_max = jnp.max(jnp.array(n_sequences))
    n_sequences = jnp.array(config.n_sequences)
    n_grid = config.n_grid

    #%%
    print(f"Sampling {n_shots_test} shots for {phis_true}.")
    t0 = time.time()
    shots, probs = sensor.sample_over_phases(
        theta, phis_true, mu, n_shots=n_shots_test, verbose=True, key=key
    )
    t1 = time.time()
    print(f"Sampling took {t1 - t0} seconds.")

    #%%
    keys = jax.random.split(key, n_trials)
    sequences = jnp.stack([select_sample_sequence(shots, key, n_sequence_max) for key in keys], axis=0)
    assert sequences.shape == (n_trials, len(phis_true), n_sequence_max, n)

    #%%
    # config.phi_center = 0.0  #jnp.pi/2/n
    phis = jnp.linspace(-jnp.pi / 2 / n + config.phi_center, jnp.pi / 2 / n + config.phi_center, 100)
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

    #%%