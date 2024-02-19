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

import itertools
import networkx as nx
from queso.sensors.tc.utils import graph_to_cz_circuit


def hardware_efficient_ansatz(c, theta, n, k):
    """ """
    for j in range(k):
        for i in range(n):
            c.ry(
                i,
                theta=theta[i, j, 0],
            )
            c.rz(
                i,
                theta=theta[i, j, 1],
            )

        for i in range(0, n - 1, 2):
            c.cz(
                i,
                i + 1,
            )
        for i in range(1, n - 1, 2):
            c.cz(
                i,
                i + 1,
            )

    for i in range(n):
        c.ry(
            i,
            theta=theta[i, k, 0],
        )
        c.rz(
            i,
            theta=theta[i, k, 1],
        )
    c.barrier_instruction()
    return c


def trapped_ion_ansatz(c, theta, n, k):
    for j in range(k):
        for i in range(n):
            c.ry(
                i,
                theta=theta[i, j, 0],
            )
            c.rz(
                i,
                theta=theta[i, j, 1],
            )

        for i in range(0, n - 1, 2):
            c.rxx(i, i + 1, theta=theta[i, j, 2])
        for i in range(1, n - 1, 2):
            c.rxx(i, i + 1, theta=theta[i, j, 3])

    for i in range(n):
        c.ry(
            i,
            theta=theta[i, k, 0],
        )
        c.rz(
            i,
            theta=theta[i, k, 1],
        )
    c.barrier_instruction()
    return c


def photonic_graph_state_ansatz(c, theta, n, k):
    # graph is bipartite, which has QFI of n^2/2 (see Shettel et al.)
    g = nx.Graph()
    nodes_a, nodes_b = [i for i in range(0, n // 2)], [i for i in range(n // 2, n)]
    g.add_nodes_from(nodes_a, bipartite=0)
    g.add_nodes_from(nodes_b, bipartite=1)
    g.add_edges_from(list(itertools.product(nodes_a, nodes_b)))

    assert n == len(g.nodes)
    c = graph_to_cz_circuit(g, c)
    for i in range(n):
        c.r(
            i,
            theta=theta[i, 0, 0],
            alpha=theta[i, 0, 1],
            phi=theta[i, 0, 2],
        )
    c.barrier_instruction()

    return c


def brick_wall_cr(c, theta, n, k):
    for j in range(k):
        for i in range(n):
            c.r(
                i,
                theta=theta[i, j, 0],
                alpha=theta[i, j, 1],
                phi=theta[i, j, 2],
            )

        for i in range(0, n - 1, 2):
            c.cr(
                i,
                i + 1,
                theta=theta[i, j, 3],
                alpha=theta[i, j, 4],
                phi=theta[i, j, 5],
            )

        for i in range(1, n - 1, 2):
            c.cr(
                i,
                i + 1,
                theta=theta[i, j, 3],
                alpha=theta[i, j, 4],
                phi=theta[i, j, 5],
            )
    c.barrier_instruction()
    return c


def brick_wall_rx_ry_cnot(c, theta, n, k):
    for j in range(k):
        for i in range(n):
            c.rx(
                i,
                theta=theta[i, j, 0],
            )
            c.ry(
                i,
                theta=theta[i, j, 1],
            )
            c.ry(
                i,
                theta=theta[i, j, 2],
            )

        for i in range(0, n - 1, 2):
            c.cnot(
                i,
                i + 1,
            )

        for i in range(1, n - 1, 2):
            c.cnot(
                i,
                i + 1,
            )
    c.barrier_instruction()
    return c


def brick_wall_cr_ancillas(c, theta, n, k, n_ancilla=1):
    for i in range(n - n_ancilla, n):
        c.r(
            i,
            theta=theta[i, 0, 0],
            alpha=theta[i, 0, 1],
            phi=theta[i, 0, 2],
        )

    for j in range(k):
        for i in range(n - n_ancilla):
            c.r(
                i,
                theta=theta[i, j, 0],
                alpha=theta[i, j, 1],
                phi=theta[i, j, 2],
            )

        for i in range(0, n - n_ancilla - 1, 2):
            c.cr(
                i,
                i + 1,
                theta=theta[i, j, 3],
                alpha=theta[i, j, 4],
                phi=theta[i, j, 5],
            )

        for i in range(1, n - n_ancilla - 1, 2):
            c.cr(
                i,
                i + 1,
                theta=theta[i, j, 3],
                alpha=theta[i, j, 4],
                phi=theta[i, j, 5],
            )
    c.barrier_instruction()
    return c


def brick_wall_cr_dephasing(c, theta, n, k, gamma=0.0):
    for j in range(k):
        for i in range(n):
            c.r(
                i,
                theta=theta[i, j, 0],
                alpha=theta[i, j, 1],
                phi=theta[i, j, 2],
            )

        for i in range(0, n - 1, 2):
            c.cr(
                i,
                i + 1,
                theta=theta[i, j, 3],
                alpha=theta[i, j, 4],
                phi=theta[i, j, 5],
            )

        for i in range(1, n - 1, 2):
            c.cr(
                i,
                i + 1,
                theta=theta[i, j, 3],
                alpha=theta[i, j, 4],
                phi=theta[i, j, 5],
            )

        for i in range(n):
            c.phasedamping(i, gamma=gamma)
    c.barrier_instruction()
    return c


def brick_wall_cr_depolarizing(c, theta, n, k, gamma=0.0):
    for j in range(k):
        for i in range(n):
            c.r(
                i,
                theta=theta[i, j, 0],
                alpha=theta[i, j, 1],
                phi=theta[i, j, 2],
            )

        for i in range(0, n - 1, 2):
            c.cr(
                i,
                i + 1,
                theta=theta[i, j, 3],
                alpha=theta[i, j, 4],
                phi=theta[i, j, 5],
            )

        for i in range(1, n - 1, 2):
            c.cr(
                i,
                i + 1,
                theta=theta[i, j, 3],
                alpha=theta[i, j, 4],
                phi=theta[i, j, 5],
            )

        for i in range(n):
            c.depolarizing(i, px=gamma / 3, py=gamma / 3, pz=gamma / 3)
    c.barrier_instruction()
    return c


def ghz_local_rotation_dephasing(c, theta, n, k, gamma):
    for i in range(0, n-1):
        c.r(
            i,
            theta=theta[i, 0, 0],
            alpha=theta[i, 0, 1],
            phi=theta[i, 0, 2]
        )

        c.r(
            i+1,
            theta=theta[i, 1, 0],
            alpha=theta[i, 1, 1],
            phi=theta[i, 1, 2]
        )

        c.cnot(
            i,
            i + 1,
        )

        c.phasedamping(i, gamma=gamma)
        c.phasedamping(i + 1, gamma=gamma)

    c.r(
        n-1,
        theta=theta[n-1, 0, 0],
        alpha=theta[n-1, 0, 1],
        phi=theta[n-1, 0, 2]
    )
    c.barrier_instruction()
    return c


def hardware_efficient_ansatz_dephasing(c, theta, n, k, gamma):
    """

    """
    for j in range(k):
        for i in range(n):
            c.ry(
                i,
                theta=theta[i, j, 0],
            )
            c.rz(
                i,
                theta=theta[i, j, 1],
            )

        for i in range(0, n - 1, 2):
            c.cz(
                i,
                i + 1,
            )
            c.phasedamping(i, gamma=gamma)
            c.phasedamping(i + 1, gamma=gamma)
        for i in range(1, n - 1, 2):
            c.cz(
                i,
                i + 1,
            )
            c.phasedamping(i, gamma=gamma)
            c.phasedamping(i + 1, gamma=gamma)

    for i in range(n):
        c.ry(
            i,
            theta=theta[i, k, 0],
        )
        c.rz(
            i,
            theta=theta[i, k, 1],
        )
    c.barrier_instruction()
    return c


def ghz_dephasing(c, theta, n, k, gamma=0.0):
    """
    A non-parameterized circuit ansatz which has a dephasing channel after each two-qubit interaction.
    """
    # c.h(0)
    # for i in range(0, n-1):
    #     c.cnot(
    #         i, i+1
    #     )
    #     c.phasedamping(i, gamma=gamma)
    #     c.phasedamping(i+1, gamma=gamma)
    # c.barrier_instruction()
    c.h(0)
    for i in range(1, n):
        c.cnot(
            0, i
        )
        # c.depolarizing(0, px=gamma / 3, py=gamma / 3, pz=gamma / 3)
        # c.depolarizing(i, px=gamma / 3, py=gamma / 3, pz=gamma / 3)
        c.phasedamping(0, gamma=gamma)
        c.phasedamping(i, gamma=gamma)
    c.barrier_instruction()
    return c
