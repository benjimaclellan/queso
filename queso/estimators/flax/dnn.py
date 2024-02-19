# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright (c) 2022-2024 Benjamin MacLellan

from typing import Sequence

import jax.nn.initializers
import jax.numpy as jnp
from flax import linen as nn


class BayesianDNNEstimator(nn.Module):
    """
    A Bayesian Deep Neural Network (DNN) Estimator implemented as a Flax module.

    This class represents a Bayesian DNN estimator, which is a type of neural network
    that can provide uncertainty estimates in addition to predictions. The network architecture
    is defined by the `nn_dims` attribute, which specifies the number of neurons in each layer.

    Attributes:
        nn_dims (Sequence[int]): A sequence of integers specifying the number of neurons in each layer of the network.

    Methods:
        __call__(self, x): Defines the computation performed at every call.
        mis
    """
    nn_dims: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for dim in self.nn_dims[:-1]:
            x = nn.relu(
                nn.Dense(dim, kernel_init=jax.nn.initializers.glorot_normal())(x)
            )
            # x = self.mish(nn.Dense(dim, kernel_init=jax.nn.initializers.glorot_normal())(x))
            # x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.Dense(self.nn_dims[-1], kernel_init=jax.nn.initializers.glorot_normal())(
            x
        )
        # x = nn.activation.softmax(x, axis=-1)
        return x

    def mish(self, x):
        return x * jnp.tanh(nn.softplus(x))
