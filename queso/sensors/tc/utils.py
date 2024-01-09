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
