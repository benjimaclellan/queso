import itertools
from collections import OrderedDict
from itertools import chain, combinations

import qutip as qt


def genuine_multipartite_entanglement(rho):
    assert isinstance(rho, qt.Qobj)
    if rho.type == "ket":
        rho = qt.ket2dm(rho)

    n = len(rho.dims[0])
    partitions = partition(list(range(n)))

    for i, (inds_a, inds_b) in enumerate(partitions):
        # for inds in (inds_a, inds_b):
        prho = qt.ptrace(rho, inds_a)
        entropy = qt.entropy_vn(prho)
        print(f"Partition {i} of {len(partitions)} | partial trace keeping {inds_a}, entropy of {entropy}.")

    return


def kbits(n):
    """ Returns a binary list of length n, where exactly half  """
    result = []
    for bits in itertools.combinations(range(n), n//2):
        s = [0] * n
        for bit in bits:
            s[bit] = 1
        if s not in result and [not i for i in s] not in result:
            result.append(s)
    return result


def partition(lst):
    """
    Function which returns all k-paritions of a list of integers - useful for computing average entanglement entropies
    """
    n = len(lst) // 2 + 1
    xs = chain(*[combinations(lst, i) for i in range(1, n)])
    pairs = (tuple(sorted([x, tuple(set(lst) - set(x))])) for x in xs)
    return list(OrderedDict.fromkeys(pairs).keys())