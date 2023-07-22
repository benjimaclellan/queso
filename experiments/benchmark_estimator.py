#%%
import time
import tqdm
import matplotlib.pyplot as plt
from itertools import cycle
import seaborn as sns
import pandas as pd
import h5py

import jax
import jax.numpy as jnp
from flax import linen as nn
from orbax.checkpoint import Checkpointer, PyTreeCheckpointHandler

from queso.estimators.flax.dnn import BayesianDNNEstimator
from queso.io import IO


#%%
def benchmark_estimator(
    io: IO,
    key: jax.random.PRNGKey = None,
    n_trials: int = 50,
    phis_inds: jnp.array = jnp.array([0]),
    n_sequences: jnp.array = jnp.round(jnp.logspace(0, 2, 10)).astype("int"),
    plot: bool = True,
):

    #%%
    # io = IO(folder="2023-07-18_fig-example-n2-k1", verbose=False)
    # io = IO(folder="2023-07-21_estimator-performance-n2-k2", verbose=False)
    # io = IO(folder="2023-07-21_circ-local-n4-k4", verbose=False)

    #%%
    hf = h5py.File(io.path.joinpath("samples.h5"), "r")
    print(hf.keys())
    shots = jnp.array(hf.get("shots_test"))
    phis = jnp.array(hf.get("phis"))
    hf.close()

    hf = h5py.File(io.path.joinpath("circ.h5"), "r")
    print(hf.keys())
    fi = jnp.array(hf.get("fi_train"))[-1]
    hf.close()

    n = shots.shape[2]
    n_phis = shots.shape[0]
    phi_range = (jnp.min(phis), jnp.max(phis))

    #%%
    ckpt_dir = io.path.joinpath("ckpts")
    ckptr = Checkpointer(PyTreeCheckpointHandler())  # A stateless object, can be created on the fly.
    restored = ckptr.restore(ckpt_dir, item=None)
    nn_dims = restored['nn_dims']

    #%%
    model = BayesianDNNEstimator(nn_dims)

    #%%
    phis_true = phis[phis_inds]
    n_sequence_max = n_sequences[-1]

    #%%
    def select_sample_sequence(shots, key):
        shot_inds = jax.random.randint(key, shape=(n_sequence_max,), minval=0, maxval=shots.shape[1])
        shots_phis = shots[phis_inds, :, :]
        sequences = shots_phis[:, shot_inds, :]
        return sequences

    if key is None:
        key = jax.random.PRNGKey(time.time_ns())
    keys = jax.random.split(key, n_trials)

    sequences = jnp.stack([select_sample_sequence(shots, key) for key in keys], axis=0)
    assert sequences.shape == (n_trials, len(phis_true), n_sequence_max, n)

    #%%
    pred = model.apply({'params': restored['params']}, sequences)
    pred = nn.activation.softmax(pred, axis=-1)
    print(pred.shape)
    assert pred.shape == (n_trials, len(phis_true), n_sequence_max, n_phis)

    #%%
    def posterior_product(pred, n_sequence):
        tmp = pred[:, :, :n_sequence, :]
        tmp = jnp.log(tmp).sum(axis=-2, keepdims=False)  # sum log posterior probs for each individual input sample
        tmp = jnp.exp(tmp - tmp.max(axis=-1, keepdims=True))  # help with underflow in normalization
        posteriors = tmp / tmp.sum(axis=-1, keepdims=True)
        return posteriors

    def estimate(posteriors, phis):
        return phis[jnp.argmax(posteriors, axis=-1)]

    @jax.jit
    def bias(phi_estimates, phis_true):
        biases = phi_estimates - phis_true[None, :, None]
        return biases

    @jax.jit
    def variance(posteriors, phi_estimates, phis):
        variances = (posteriors * jnp.power(phi_estimates[:, :, :, None] - phis[None, None, None, :], 2)).sum(axis=-1)
        return variances

    posteriors = jnp.stack([posterior_product(pred, n_sequence) for n_sequence in n_sequences], axis=2)
    assert posteriors.shape == (n_trials, len(phis_true), len(n_sequences), n_phis)

    phi_estimates = estimate(posteriors, phis)
    biases = bias(phi_estimates, phis_true)
    variances = variance(posteriors, phi_estimates, phis)

    #%% plot updated posterior distribution for n_trials different sequence samples
    if plot:
        fig, axs = plt.subplots(nrows=n_trials, ncols=phis_inds.shape[0], figsize=(10.0, 1.5 * n_trials), sharex=True, sharey=True)
        colors = sns.color_palette('crest', n_colors=n_sequences.shape[0])
        markers = ["o", "D", 's', "v", "^", "<", ">", ]

        for k in range(phis_inds.shape[0]):
            for j in range(n_trials):
                for i, n_sequence in enumerate(n_sequences):
                    ax = axs[j, k]
                    ax.axvline(phis_true[k], color='black', ls='-', alpha=1.0)
                    ax.axvline(phi_estimates[j, k, -1], color='red', ls='-', alpha=1.0)
                    p = posteriors[j, k, i, :]
                    ax.plot(
                        jnp.linspace(phi_range[0], phi_range[1], n_phis),
                        p / jnp.max(p),
                        ls=':',
                        marker=markers[i % len(markers)],
                        color=colors[i],
                        alpha=(i+1) / len(n_sequences),
                        markersize=3,
                )
        io.save_figure(fig, 'trials_n_sequences.pdf')
        del fig

    #%% plot posterior, bias, and variance for one phase
    for k in range(phis_inds.shape[0]):
        fig, axs = plt.subplots(nrows=3, figsize=(6.5, 6.0))
        colors = sns.color_palette('crest', n_colors=n_sequences.shape[0])
        markers = ["o", "D", 's', "v", "^", "<", ">", ]

        for i, n_sequence in enumerate(n_sequences):
            ax = axs[0]
            ax.axvline(phis_true[k], color=colors[0], ls='--', alpha=0.7)
            p = posteriors[1, k, i, :]
            ax.plot(
                jnp.linspace(phi_range[0], phi_range[1], n_phis),
                p / jnp.max(p),
                ls=':',
                marker=markers[i % len(markers)],
                color=colors[i],
                # alpha=(0.1 + i / len(n_sequences) * 0.8),
                alpha=(i+1) / len(n_sequences),
                markersize=3,
                label=r"$\phi_{true}=$" + f"{phis_true[k] / jnp.pi:0.2f}$\pi$",
            )

        line_kwargs = dict(color='grey', alpha=0.6, ls='--')
        ax = axs[1]
        ax.errorbar(
            n_sequences,
            biases[:, k, :].mean(axis=0),
            xerr=None,
            yerr=jnp.var(biases[:, k, :], axis=0),
            color=colors[0],
            ls=':',
            marker=markers[0],
        )
        ax.axhline(0, **line_kwargs)
        ax.set(xscale="log")

        ax = axs[2]
        ax.plot(
            n_sequences,
            variances[:, k, :].mean(axis=0),
            color=colors[0],
            ls=':',
            marker=markers[0],
        )
        ax.plot(n_sequences, 1/(n_sequences * fi), label='CRB', **line_kwargs)
        ax.plot(n_sequences, 1/(n_sequences * n), label='SQL', **dict(color='black', alpha=0.8, ls=':'))
        ax.plot(n_sequences, 1/(n_sequences * n**2), label='HL', **dict(color='black', alpha=0.8, ls=':'))
        ax.set(xscale="log", yscale='log')

        axs[0].set(xlabel="$\phi_j$", ylabel=r"p($\phi_j | \vec{s}$)")
        axs[1].set(xlabel="Sequence length, $m$", ylabel=r"Bias, $\langle \hat{\varphi} - \varphi \rangle$")
        axs[2].set(xlabel="Sequence length, $m$", ylabel=r"Variance, $\langle \Delta^2 \hat{\phi} \rangle$")

        io.save_figure(fig, filename=f"bias-variance/{k}_{phis_true[k].item()}.png")
        fig.tight_layout()
        plt.show()

