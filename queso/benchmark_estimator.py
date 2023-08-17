#%%
import time
import tqdm
import matplotlib.pyplot as plt
from itertools import cycle
import seaborn as sns
import pandas as pd
import h5py
import argparse

import jax
import jax.numpy as jnp
from flax import linen as nn
from orbax.checkpoint import Checkpointer, PyTreeCheckpointHandler

from queso.estimators.flax.dnn import BayesianDNNEstimator
from queso.io import IO
from queso.configs import Configuration


#%%
def benchmark_estimator(
    io: IO,
    config: Configuration,
    key: jax.random.PRNGKey = None,
    plot: bool = False,
):

    #%%
    n_trials = config.n_trials
    n_sequences = jnp.array(config.n_sequences)
    n_grid = config.n_grid

    #%%
    hf = h5py.File(io.path.joinpath("train_samples.h5"), "r")
    print(hf.keys())
    phis = jnp.array(hf.get("phis"))
    hf.close()

    hf = h5py.File(io.path.joinpath("test_samples.h5"), "r")
    print(hf.keys())
    shots_test = jnp.array(hf.get("shots_test"))
    shots = shots_test
    phis_test = jnp.array(hf.get("phis_test"))
    hf.close()

    hf = h5py.File(io.path.joinpath("circ.h5"), "r")
    print(hf.keys())
    fi = jnp.array(hf.get("fi_train"))[-1]
    hf.close()

    hf = h5py.File(io.path.joinpath("nn.h5"), "r")
    print(hf.keys())
    grid = jnp.array(hf.get("grid"))
    hf.close()

    n = config.n
    n_phis = config.n_phis
    phi_range = (jnp.min(phis), jnp.max(phis))

    #%%
    ckpt_dir = io.path.joinpath("ckpts")
    ckptr = Checkpointer(PyTreeCheckpointHandler())  # A stateless object, can be created on the fly.
    restored = ckptr.restore(ckpt_dir, item=None)
    nn_dims = restored['nn_dims']

    #%%
    model = BayesianDNNEstimator(nn_dims)

    #%%
    phis_true = phis_test
    n_sequence_max = jnp.max(n_sequences)

    #%%
    def select_sample_sequence(shots, key):
        shot_inds = jax.random.randint(key, shape=(n_sequence_max,), minval=0, maxval=shots.shape[1])
        # shots_phis = shots[phis_inds, :, :, :]
        sequences = shots[:, shot_inds, :]
        return sequences

    if key is None:
        key = jax.random.PRNGKey(time.time_ns())
    keys = jax.random.split(key, n_trials)

    sequences = jnp.stack([select_sample_sequence(shots, key) for key in keys], axis=0)
    assert sequences.shape == (n_trials, len(phis_true), n_sequence_max, n)

    #%%
    pred = model.apply({'params': restored['params']}, sequences)
    # T =
    # pred = nn.activation.softmax(pred / T, axis=-1)  # use temperature scaling to smooth
    pred = nn.activation.softmax(pred, axis=-1)
    print(pred.shape)
    assert pred.shape == (n_trials, len(phis_true), n_sequence_max, n_grid)

    #%%
    def posterior_product(pred, n_sequence):
        # shape is of [n_trials, n_phis_true, n_sequences, n_grid|n_phis]
        tmp = pred[:, :, :n_sequence, :]
        tmp = jnp.log(tmp).sum(axis=-2, keepdims=False)  # sum log posterior probs for each individual input sample
        tmp = jnp.exp(tmp - tmp.max(axis=-1, keepdims=True))  # help with underflow in normalization
        posteriors = tmp / tmp.sum(axis=-1, keepdims=True)
        return posteriors

    def estimate(posteriors, phis):
        return phis[jnp.argmax(posteriors, axis=-1)]

    @jax.jit
    def bias(phis_estimates, phis_true):
        # bias in Euclidean space
        biases = phis_estimates - phis_true[None, :, None]

        # bias on a circular manifold
        # vecs_true = jnp.exp(1j * phis_true[None, :, None])
        # vecs_est = jnp.exp(1j * phis_estimates)
        # biases = jnp.arccos(vecs_true.real * vecs_est.real + vecs_true.imag * vecs_est.imag)
        return biases

    @jax.jit
    def variance(posteriors, phis_estimates, phis):
        # over a euclidean space
        variances = (posteriors * jnp.power(phis_estimates[:, :, :, None] - grid[None, None, None, :], 2)).sum(axis=-1)

        # over a circular space
        # vecs_est = jnp.exp(1j * phis_estimates[:, :, :, None])
        # vecs_out = jnp.exp(1j * phis[None, None, None, :])
        # tmp = jnp.clip(vecs_out.real * vecs_est.real + vecs_out.imag * vecs_est.imag, -1, 1)
        # diff = jnp.arccos(tmp)
        # # print(jnp.count_nonzero(jnp.isnan(diff)))
        # variances = (posteriors * jnp.power(diff, 2)).sum(axis=-1)
        return variances

    posteriors = jnp.stack([posterior_product(pred, n_sequence) for n_sequence in n_sequences], axis=2)
    assert posteriors.shape == (n_trials, len(phis_true), len(n_sequences), n_grid)

    phis_estimates = estimate(posteriors, phis)
    biases = bias(phis_estimates, phis_true)
    variances = variance(posteriors, phis_estimates, phis)

    #%%
    hf = h5py.File(io.path.joinpath("estimates.h5"), "w")
    hf.create_dataset("phis_estimates", data=phis_estimates)
    hf.create_dataset("phis_true", data=phis_true)
    hf.create_dataset("n_sequences", data=n_sequences)
    hf.create_dataset("posteriors", data=posteriors)
    hf.create_dataset("biases", data=biases)
    hf.create_dataset("variances", data=variances)
    hf.create_dataset("phis", data=phis)
    hf.close()

    #%% plot updated posterior distribution for n_trials different sequence samples
    ncols = phis_test.size
    nrows = 8
    if plot:
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3.5*ncols, 1.5 * nrows),
                                sharex=True, sharey=True, gridspec_kw=dict(hspace=0, wspace=0))
        colors = sns.color_palette('crest', n_colors=n_sequences.shape[0])

        for k in range(ncols):
            z = k * (phis_true.shape[0] // ncols)
            print(f"Closest grid point is {jnp.min(jnp.abs(phis - phis_true[z]))}")
            for j in range(nrows):
                for i, n_sequence in enumerate(n_sequences):
                    # print(k, j, i)
                    ax = axs[j, k]
                    ax.axvline(phis_estimates[j, z, -1], color='red', ls='-', lw=1, alpha=1.0)
                    ax.axvline(phis_true[z], color='gray', ls='--', lw=1, alpha=0.6)
                    p = posteriors[j, z, i, :]
                    ax.plot(
                        grid,
                        p / jnp.max(p),
                        ls='-',
                        lw=1,
                        color=colors[i],
                        alpha=(i+1) / len(n_sequences),
                    )
                    b = phis_true[z] - phis_estimates[j, z, -1]
                    ax.annotate(text=f"{b:2.6f}",
                                xy=(0.9, 0.9),
                                xycoords='axes fraction',
                                **dict(ha='right', va='top'))
                    ax.set(xticks=[], yticks=[])

        io.save_figure(fig, 'trials_n_sequences_m_phases.pdf')
        del fig

    #%%
    if plot:
        ncols = 1
        nrows = n_trials

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3.5 * ncols, 1.5 * nrows), sharex=True, sharey=True,
                                gridspec_kw=dict(hspace=0, wspace=0))
        colors = sns.color_palette('crest', n_colors=n_sequences.shape[0])
        k = phis_true.shape[0] // 2
        print(f"Closest grid point is {jnp.min(jnp.abs(phis - phis_true[z]))}")
        for j in range(nrows):
            for i, n_sequence in enumerate(n_sequences):
                # print(k, j, i)
                ax = axs[j]
                ax.axvline(phis_estimates[j, k, -1], color='red', ls='-', lw=1, alpha=1.0)
                ax.axvline(phis_true[k], color='gray', ls='--', lw=1, alpha=0.6)
                p = posteriors[j, k, i, :]
                ax.plot(
                    grid,
                    p / jnp.max(p),
                    ls='-',
                    lw=1,
                    color=colors[i],
                    alpha=(i + 1) / len(n_sequences),
                )
                b = phis_true[k] - phis_estimates[j, k, -1]
                ax.annotate(text=f"{b:2.6f}",
                            xy=(0.9, 0.9),
                            xycoords='axes fraction',
                            **dict(ha='right', va='top'))
                ax.set(xticks=[], yticks=[])

        io.save_figure(fig, 'all_trials_one_phase.pdf')
        del fig


    #%% plot posterior, bias, and variance for one phase
    for k in range(phis_true.shape[0]):
        fig, axs = plt.subplots(nrows=3, figsize=(6.5, 6.0))
        colors = sns.color_palette('crest', n_colors=n_sequences.shape[0])
        markers = ["o", "D", 's', "v", "^", "<", ">", ]
        axs[0].axvline(phis_true[k], color='black', ls='--', alpha=1.0, lw=2)

        for i, n_sequence in enumerate(n_sequences):
            ax = axs[0]
            p = posteriors[1, k, i, :]
            ax.plot(
                grid,
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


#%%
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="tmp")
    args = parser.parse_args()
    folder = args.folder

    io = IO(folder=f"{folder}")
    print(io)
    config = Configuration.from_yaml(io.path.joinpath('config.yaml'))
    key = jax.random.PRNGKey(config.seed)
    print(f"Benchmarking NN: {folder} | Devices {jax.devices()} | Full path {io.path}")
    print(f"Config: {config}")
    benchmark_estimator(io, config, key, plot=True)
