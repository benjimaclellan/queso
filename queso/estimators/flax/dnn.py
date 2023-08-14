from typing import Sequence

import jax.nn.initializers
import jax.numpy as jnp
from flax import linen as nn


class BayesianDNNEstimator(nn.Module):
    nn_dims: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for dim in self.nn_dims[:-1]:
            x = nn.relu(nn.Dense(dim, kernel_init=jax.nn.initializers.glorot_normal())(x))
            # x = self.mish(nn.Dense(dim, kernel_init=jax.nn.initializers.glorot_normal())(x))
            # x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.Dense(self.nn_dims[-1], kernel_init=jax.nn.initializers.glorot_normal())(x)
        # x = nn.activation.softmax(x, axis=-1)
        return x

    def mish(self, x):
        return x * jnp.tanh(nn.softplus(x))
