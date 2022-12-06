import jax
import jax.numpy as np
from jax.config import config

from qsense.functions import dagger, compile

config.update("jax_enable_x64", True)


def qfim(params, circuit, ket_i, keys):
    """
    Calculates the Quantum Fischer information matrix, with respect to the parameter keys provided in `keys`.
    :param params:
    :param circuit:
    :param keys:
    :return:
    """
    # parameters to compute QFI for must be complex datatype in order to automatically differentiate
    tunable_params = {key: np.complex128(params[key]) for key in keys}

    def fket(tunable_params):
        params.update(tunable_params)
        u = compile(params, circuit)
        return u @ ket_i

    fdket = jax.jacrev(fket, holomorphic=True)
    ket = fket(tunable_params)
    dket = fdket(tunable_params)

    # flatten parameters to a single list
    p = [(key, i) for key in keys for i in range(len(params[key]))]

    f = []
    for i, (key_a, a) in enumerate(p):
        fa = []
        for j, (key_b, b) in enumerate(p):
            da = dket[key_a][:, :, a]
            db = dket[key_b][:, :, b]
            f_ab = 4 * np.real(
                dagger(da) @ db - (dagger(da) @ ket) * (dagger(ket) @ db)
            )
            fa.append(f_ab.squeeze())
        f.append(fa)
    f = np.array(f)
    return f
