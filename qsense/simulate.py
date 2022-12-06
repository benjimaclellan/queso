import jax
import jax.numpy as np
from functools import partial
import time
import seaborn as sns
import matplotlib.pyplot as plt

from qsense.unitaries import *
from qsense.states import *
from qsense.utils import tensor, sum, prod

import numpy


def initialize(circuit):
    params = {}
    rng_key = jax.random.PRNGKey(time.time_ns())
    for layer in circuit:
        for (u, key) in layer:
            if (unitary_info[u]["bounds"] is not None) and (unitary_info[u]["m"] != 0):
                m = unitary_info[u]["m"]
                (low, high) = unitary_info[u]["bounds"]
                if key not in params.keys():
                    rng_key, rng_subkey = jax.random.split(rng_key)
                    params[key] = jax.random.uniform(key=rng_subkey, shape=[m], minval=low, maxval=high)
    return params


# @partial(jax.jit, static_argnums=(1,))
def compile(params, circuit):
    us = []
    for layer in circuit:
        us.append(
            tensor(
                [
                    u(*params[key]) if (key in params.keys()) else u()
                    for (u, key) in layer
                ]
            )
        )
    # for u in us:
    #     print("\n", u)
    #     fig, axs = plt.subplots(1, 2)
    #     sns.heatmap(np.real(u), ax=axs[0])
    #     sns.heatmap(np.imag(u), ax=axs[1])
    #     plt.show()
    u = prod(us)
    # print("compiling")
    # print(u)
    # fig, axs = plt.subplots(1, 2)
    # sns.heatmap(np.real(u), ax=axs[0])
    # sns.heatmap(np.imag(u), ax=axs[1])
    # plt.show()
    return u
