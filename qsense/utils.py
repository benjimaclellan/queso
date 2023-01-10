import jax.numpy as np
import functools
import operator
import itertools
from collections import OrderedDict
from itertools import chain, combinations


def sum(args):
    return functools.reduce(operator.add, args)


def prod(args):
    return functools.reduce(operator.matmul, args)


def tensor(args):
    return functools.reduce(np.kron, args)


""" 
Function which returns all k-paritions of a list of integers - useful for computing average entanglement entropies
"""

def kbits(n):
    """ Returns a binary list of length n, where exactly half  """
    result = []
    for bits in itertools.combinations(range(n), n//2):
        s = [0] * n
        for bit in bits:
            s[bit] = 1
        print(s)
        print([not i for i in s])
        if s not in result and [not i for i in s] not in result:
            result.append(s)
    return result


def partition(lst):
    n = len(lst) // 2 + 1
    xs = chain(*[combinations(lst, i) for i in range(1, n)])
    pairs = (tuple(sorted([x, tuple(set(lst) - set(x))])) for x in xs)
    return list(OrderedDict.fromkeys(pairs).keys())
