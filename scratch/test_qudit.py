if __name__ == "__main__":

    d = 3
    n = 3

    params = [0.0, 1.0, 2.0]
    u = np.identity(d)

    for (i, j), param in zip(itertools.combinations(range(d), 2), params):
        print(i, j, param)
        rot = np.identity(d)

        rot = rot.at[i, j].set(-np.sin(param))
        rot = rot.at[j, i].set(np.sin(param))
        rot = rot.at[i, i].set(np.cos(param))
        rot = rot.at[j, j].set(np.cos(param))
        print(rot)

        u = rot @ u
    print(u)
    print(dagger(u) @ u)