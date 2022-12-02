def local_layer(params, n, d=2):
    u = tensor([phase(theta) for theta in params])
    return u


def local_phase(phi, ket, n=1, d=2):
    u = tensor(n * [phase(phi)])
    ket = u @ ket
    return ket
