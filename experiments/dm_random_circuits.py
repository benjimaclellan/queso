import copy
from functools import partial
import time

import pandas as pd
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import numpy as np
from functools import reduce

import tensorcircuit as tc
from tensorcircuit.quantum import sample_bin2int, sample_int2bin
import jax
import jax.numpy as jnp

backend = tc.set_backend("jax")
tc.set_dtype("complex128")
tc.set_contractor("greedy")  # “auto”, “greedy”, “branch”, “plain”, “tng”, “custom”

#%%
n = 6
ks = jnp.arange(1, 10)
ps = (jnp.logspace(0, 1.0, 10) - 1)/10

#%%
# measure = jnp.stack(
#         [jax.random.choice(key, jnp.array([0, 1]), shape=[n, max(ks)], p=jnp.array([1.0-p, p])) for p in ps], axis=0
# )
# m0 = jnp.array([jnp.array([[1, 0], [0, 1]]), jnp.array([[1, 0], [0, 1]])])
# m1 = jnp.array([jnp.array([[1, 0], [0, 0]]), jnp.array([[0, 0], [0, 1]])])
# kraus = (
#         (1 - measure[:, :, :, None, None, None]) * m0[None, None, None, :, :, :]
#         + measure[:, :, :, None, None, None] * m1[None, None, None, :, :, :]
# )
#
# c = tc.Circuit(1)
# c.h(0)
# c.general_kraus(
#                 jnp.array([jnp.array([[1.0, 0], [0, 0]]), jnp.array([[0, 0], [0, 1]])]),
#                 # jnp.array([jnp.array([[1.0, 0], [0, 1.0]]), jnp.array([[1.0, 0], [0, 1.0]])]),
#                 0,
#                 name="measure",
#             )
# print(c.state())


#%%
sigma_z = jnp.array([[1, 0], [0, -1]])
sigma_x = jnp.array([[0, 1], [1, 0]])
# oper = reduce(jnp.kron, n * [sigma_z])
# oper = reduce(jnp.kron, n * [sigma_z])

# psi0 = 1 / jnp.sqrt(2) * (reduce(jnp.kron, n * [jnp.array([1, 0])]) + reduce(jnp.kron, n * [jnp.array([0, 1])]))[:, None]
# fisher_info = 4 * (
#                 (psi0.conj().T @ oper @ oper @ psi0)
#                 - (psi0.conj().T @ oper @ psi0)**2
#         )
#
#
# def state(phi):
#     # rx = jnp.array([[jnp.cos(phi/2), -1j * jnp.sin(phi/2)], [-1j * jnp.sin(phi/2), jnp.cos(phi/2)]])
#     rz = jnp.array([[1, 0], [0, jnp.exp(1j * phi)]])
#     return reduce(jnp.kron, n * [rz]) @ psi0
#
#
# phi = jnp.array(1.0)
# psi = state(phi)
# dpsi = jax.jacrev(state, argnums=0, holomorphic=True)(phi.astype("complex64"))
# fisher_info = (
#         4 * jnp.real(
#             dpsi.conj().T @ dpsi
#             - jnp.abs(dpsi.conj().T @ psi)**2
#         ).squeeze()
# )
# print(fisher_info)

#%%
def simulate_circuit(kraus, u_haar):
    ee = []
    fi = []
    c = tc.Circuit(n)
    for j, k in enumerate(ks):
        # print(f"repeat = {repeat} | p={p:0.3f} | k={k}")
        for i in range(0, n-1, 2):
            c.unitary(i, i + 1, unitary=tc.gates.Gate(u_haar[i, j, :, :, :]))
        for i in range(1, n-1, 2):
            c.unitary(i, i + 1, unitary=tc.gates.Gate(u_haar[i, j, :, :, :, :]))

        for i in range(n):
            c.general_kraus(
                kraus[i, j, :, :, :],  # kraus operators either performing measurement or identity
                i,
                name="measure",
            )

        # entropy
        psi0 = c.state()
        rho_A = tc.quantum.reduced_density_matrix(
            psi0, [x for x in range(n//2)]
        )
        entropy = tc.quantum.entropy(rho_A)
        ee.append(entropy)


        def state(phi):
            # rx = jnp.array([[jnp.cos(phi/2), -1j * jnp.sin(phi/2)], [-1j * jnp.sin(phi/2), jnp.cos(phi/2)]])
            rz = jnp.array([[1, 0], [0, jnp.exp(1j * phi)]])
            return reduce(jnp.kron, n * [rz]) @ psi0

        phi = jnp.array(1.0)
        psi = state(phi)
        dpsi = jax.jacrev(state, argnums=0, holomorphic=True)(phi.astype("complex64"))
        fisher_info = (
                4 * jnp.real(
            dpsi.conj().T @ dpsi
            - jnp.abs(dpsi.conj().T @ psi) ** 2
        ).squeeze()
        )
        fi.append(fisher_info.real.squeeze())

    return jnp.array(ee), jnp.array(fi)


simulate_circuit_jit = jax.jit(simulate_circuit)



#%%
n_repeat = 1000
key = jax.random.PRNGKey(1234)
entropy = np.zeros(shape=[len(ps), len(ks), n_repeat])
fisher_info = np.zeros(shape=[len(ps), len(ks), n_repeat])
for repeat in range(n_repeat):
    u_haar = jnp.stack(
        [jnp.stack([tc.gates.random_two_qubit_gate().tensor for i in range(n)], axis=0) for j in range(max(ks))], axis=1
    )
    for l, p in enumerate(ps):
        print(f"repeat = {repeat} | p = {p}")
        key, subkey = jax.random.split(key)
        measure = jax.random.choice(subkey, jnp.array([0, 1]), shape=[n, max(ks)], p=jnp.array([1.0 - p, p]))
        m0 = jnp.array([jnp.array([[1, 0], [0, 1]]), jnp.array([[1, 0], [0, 1]])])
        m1 = jnp.array([jnp.array([[1, 0], [0, 0]]), jnp.array([[0, 0], [0, 1]])])
        kraus = (
                (1 - measure[:, :, None, None, None]) * m0[None, None, :, :, :]
                + measure[:, :, None, None, None] * m1[None, None, :, :, :]
        )
        ee, fi = simulate_circuit_jit(kraus, u_haar)
        entropy[l, :, repeat] = ee
        fisher_info[l, :, repeat] = fi

entropy_avg = entropy.mean(axis=-1)
entropy_std = entropy.std(axis=-1)
fisher_info_avg = fisher_info.mean(axis=-1)
fisher_info_std = fisher_info.std(axis=-1)


#%%
fig, ax = plt.subplots()
colors = sns.color_palette('mako', n_colors=len(ps))
for l, p in enumerate(ps):
    ax.plot(ks, entropy_avg[l, :], label=f"$p=${p:0.3f}", color=colors[l], ls='-', lw=1)
ax.set(xlabel='Circuit time, $k$', ylabel='Entropy of entanglement')
ax.legend()
fig.show()

#%%
fig, ax = plt.subplots()
colors = sns.color_palette('mako', n_colors=len(ps))
for l, p in enumerate(ps):
    ax.plot(ks, fisher_info_avg[l, :], label=f"$p=${p:0.3f}", color=colors[l], ls='-', lw=1)
    ax.fill_between(
        ks,
        # jnp.min(fisher_info, axis=-1)[l, :],
        # jnp.max(fisher_info, axis=-1)[l, :],
        fisher_info_avg[l, :] - fisher_info_std[l, :],
        fisher_info_avg[l, :] + fisher_info_std[l, :],
        color=colors[l], ls='-', lw=1, alpha=0.04
    )

ax.set(xlabel='Circuit time, $k$', ylabel='Fisher Information')
# ax.legend()
fig.show()

#%%