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
import tensorcircuit as tc
from tensorcircuit.quantum import sample_bin2int, sample_int2bin
from jax import numpy as jnp
import networkx as nx


def shots_to_counts(shots):
    # shots = jnp.array(list(zip(*shots))[0]).astype("int8")
    basis, count = jnp.unique(shots, return_counts=True, axis=0)
    return {
        "".join([str(j) for j in basis[i]]): count[i].item() for i in range(len(count))
    }


def counts_to_list(counts, n):
    bin_str = ["".join(p) for p in itertools.product("01", repeat=n)]
    return [counts.get(b, 0) for b in bin_str]


def graph_to_cz_circuit(g: nx.Graph, c: tc.Circuit):
    assert (
        len(g.nodes) == c._nqubits
    ), "Graph nodes do not match number of circuit qubits."
    for node in g.nodes:
        c.h(node)
    for edge in g.edges:
        c.cz(edge[0], edge[1])
    return c
