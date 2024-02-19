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

from typing import Optional

import torch
from torch.utils import data as data


class SensorDataset(data.Dataset):
    def __init__(self, shots: torch.Tensor, phis: torch.Tensor):
        self.shots = shots
        self.phis = phis

        self.n_phis = shots.shape[0]
        self.n_shots = shots.shape[1]
        self.n = shots.shape[2]

        self.shuffle = True

    def __getitem__(self, shots_idx):
        shots = self.shots[:, shots_idx, :]
        phis = self.phis
        if self.shuffle:
            inds = torch.randperm(self.n_phis)
            phis = phis[inds]
            shots = shots[inds, :, :]
        return shots, phis

    def __len__(self):
        return self.n_phis


class SensorSampler(data.Sampler):
    def __init__(
        self,
        data_source: SensorDataset,
        replacement: bool = False,
        n_samples: Optional[int] = None,
        generator=None,
    ):
        super().__init__(data_source)
        self.dataset = data_source
        self.replacement = replacement
        self._n_samples = n_samples
        self.generator = generator

    @property
    def n_samples(self) -> int:
        return self._n_samples

    def __iter__(self):
        n_shots = self.dataset.n_shots
        if self.replacement:
            inds = torch.randint(
                0, n_shots, size=(self.n_samples,)
            )  # todo: sample different rows for each phase
        else:
            raise NotImplementedError
        yield inds

    def __len__(self) -> int:
        return self.n_samples
