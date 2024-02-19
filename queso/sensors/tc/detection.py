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

def local_r(c, mu, n, k):
    for i in range(n):
        c.r(
            i,
            theta=mu[i, 0],
            alpha=mu[i, 1],
            phi=mu[i, 2],
        )
    c.barrier_instruction()
    return c


def computational_bases(c, mu, n, k):
    return c


def hadamard_bases(c, mu, n, k):
    for i in range(n):
        c.h(i)
    return c


def local_rx_ry_ry(c, mu, n, k):
    for i in range(n):
        c.rx(i, theta=mu[i, 0])
        c.ry(
            i,
            theta=mu[i, 1],
        )
        c.ry(
            i,
            theta=mu[i, 2],
        )
    c.barrier_instruction()
    return c
