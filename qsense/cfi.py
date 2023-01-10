import jax
import jax.numpy as np
from jax.config import config

from qsense.functions import dagger, compile

config.update("jax_enable_x64", True)


def cfim(params, probe_circ, interaction_circ, measure_circ, ket_i, proj_mu, keys):
    # parameters to compute CFI for must be complex datatype in order to automatically differentiate
    # tunable_params = {key: np.complex128(params[key]) for key in keys}
    tunable_params = {key: params[key] for key in keys}

    def fp(tunable_params):
        params.update(tunable_params)
        ket = ket_i
        ket = compile(params, probe_circ) @ ket
        ket = compile(params, interaction_circ) @ ket
        ket = compile(params, measure_circ) @ ket
        povm = np.abs(ket) ** 2
        return povm

    fdp = jax.jacrev(fp, holomorphic=False)
    p = fp(tunable_params)
    dp = fdp(tunable_params)

    # flatten parameters to a single list
    plist = [(key, i) for key in keys for i in range(len(params[key]))]

    f = []
    for i, (key_a, a) in enumerate(plist):
        fa = []
        for j, (key_b, b) in enumerate(plist):
            da = dp[key_a][:, 0, a]
            db = dp[key_b][:, 0, b]
            f_ab = np.sum(da * db / p[:, 0])
            fa.append(f_ab.squeeze())
        f.append(fa)
    f = np.array(f)
    return f
