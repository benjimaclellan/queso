from typing import Sequence

from flax import linen as nn


class BayesianDNNEstimator(nn.Module):
    nn_dims: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for dim in self.nn_dims[:-1]:
            x = nn.relu(nn.Dense(dim)(x))
        x = nn.Dense(self.nn_dims[-1])(x)
        # x = nn.activation.softmax(x, axis=-1)
        return x