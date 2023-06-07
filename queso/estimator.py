from typing import Sequence

from flax import linen as nn


class RegressionEstimator(nn.Module):
    # todo: build FF-NN for regression
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for features in self.features[:-1]:
            x = nn.relu(nn.Dense(features=features)(x))
        x = nn.Dense(self.features[-1])(x)
        return x
