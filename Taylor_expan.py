import numpy as np
def Taylor_expan(x, power):
    if power == 1:
        X = np.vstack([np.ones((1, x.shape[1])), x])
    elif power == 2:
        x2 = []
        xt1 = []
        xt2 = x
        n = x.shape[0]
        for i in range(n):
            xt1 = xt2 * np.repeat(xt2[0:1, :], xt2.shape[0], axis=0)
            xt2 = xt2[1:, :]
            x2.append(xt1)
        x2 = np.vstack(x2)
        X = np.vstack([np.ones((1, x.shape[1])), x, x2])
    elif power == 3:
        x2 = []
        xt1 = []
        xt2 = x
        n = x.shape[0]
        for i in range(n):
            xt1 = xt2 * np.repeat(xt2[0:1, :], xt2.shape[0], axis=0)
            xt2 = xt2[1:, :]
            x2.append(xt1)
        x2 = np.vstack(x2)

        x3 = []
        xtt1 = []
        xtt2 = x2
        xtt3 = x
        for i in range(n):
            xtt1 = xtt2 * np.repeat(xtt3[0:1, :], xtt2.shape[0], axis=0)
            xtt2 = xtt2[n + 1 - i:, :]
            x3.append(xtt1)
        x3 = np.vstack(x3)
        X = np.vstack([np.ones((1, x.shape[1])), x, x2, x3])
    return X
