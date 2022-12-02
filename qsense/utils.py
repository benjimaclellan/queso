import jax.numpy as np
import functools
import operator


def sum(args):
    return functools.reduce(operator.add, args)


def prod(args):
    return functools.reduce(operator.matmul, args)


def tensor(args):
    return functools.reduce(np.kron, args)
