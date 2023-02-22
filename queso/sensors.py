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
        c.r(j, theta=params[3 * j, k], alpha=params[3 * j + 1, k + 1], phi=params[3 * j + 2, k])

    return c
