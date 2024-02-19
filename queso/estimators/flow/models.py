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

import normflows.distributions
import torch
from torch import nn
import normflows as nf

import einops


class Dense(nn.Module):
    def __init__(self, layer_widths):
        super(Dense, self).__init__()
        layers = []
        for i in range(len(layer_widths) - 1):
            layers.append(nn.Linear(layer_widths[i], layer_widths[i + 1]))
            if i < len(layer_widths) - 2:
                layers.append(nn.LeakyReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class RNN(nn.Module):
    def __init__(
        self, dim_input: int, dim_hidden: int, dim_output: int, num_layers: int
    ):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            dim_input, dim_hidden, num_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(dim_hidden, dim_output)

    def forward(self, x):
        out, _ = self.rnn(x, None)
        out = self.fc(out[:, -1, :])  # Take the output from the last time step
        return out


class Flow(normflows.NormalizingFlow):
    def __init__(
        self,
        base: nf.distributions.BaseDistribution,
        num_layers: int = 4,
    ):
        flows = []
        for i in range(num_layers):
            flows.append(nf.flows.Planar(shape=[1], act="leaky_relu"))
            # flows.append(nf.flows.AffineConstFlow(shape=[1]))
            # flows.append(nf.flows.Radial(shape=[1], z_0=None))
            # flows.append(nf.flows.Squeeze())
            # nf.flows.CCAffineConst(shape=[1], num_classes=1)

        super().__init__(base, flows)


# class Estimator(nn.Module):
#     def __init__(
#             self,
#             encoder: nn.Module,
#             flow: nn.Module
#     ):
#         super(Estimator, self).__init__()
#         self.encoder = encoder
#         self.flow = flow
#
#     def forward(self, x):
#         latent = self.encoder(x)
#
#         # sample base dist.
#         eps = self.flow.base.sample(1000).squeeze()
#         z = latent[:, 0] + torch.exp(latent[:, 1]) * eps
#
#         return self.model(x)
