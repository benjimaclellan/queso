import tensorcircuit as tc


def cnot_2local_dephased_ansatz(params, phi, gamma, n, k):
    c = tc.DMCircuit(n)
    for i in range(k):
        for j in range(n):
            c.r(
                j,
                theta=params[3 * j, i],
                alpha=params[3 * j + 1, i],
                phi=params[3 * j + 2, i],
            )

        for j in range(1, n, 2):
            c.cnot(j - 1, j)

        for j in range(2, n, 2):
            c.cnot(j - 1, j)

        for j in range(n):
            c.phasedamping(j, gamma=gamma)

    # interaction
    for j in range(n):
        c.rz(j, theta=phi)

    # measurement
    for j in range(n):
        c.r(
            j,
            theta=params[3 * j, k],
            alpha=params[3 * j + 1, k],
            phi=params[3 * j + 2, k],
        )

    return c


def cnot_2local_ansatz(params, phi, n, k):
    c = tc.Circuit(n)
    for i in range(k):
        for j in range(n):
            c.r(
                j,
                theta=params[3 * j, i],
                alpha=params[3 * j + 1, i],
                phi=params[3 * j + 2, i],
            )

        for j in range(1, n, 2):
            c.cnot(j - 1, j)

        for j in range(2, n, 2):
            c.cnot(j - 1, j)

    # interaction
    for j in range(n):
        c.rz(j, theta=phi)

    # measurement
    for j in range(n):
        c.r(
            j,
            theta=params[3 * j, k],
            alpha=params[3 * j + 1, k],
            phi=params[3 * j + 2, k],
        )

    return c


def build(name, n, k):
    if name == "cnot_2local_dephased_ansatz":
        circuit, shape = cnot_2local_dephased_ansatz, [3 * n, k + 1]
    elif name == "cnot_2local_ansatz":
        circuit, shape = cnot_2local_ansatz, [3 * n, k + 1]
    else:
        raise NotImplementedError("Not a valid probe state name")
    return circuit, shape
