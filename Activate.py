def Activate(x, type):
    if type == 1:  # RELU
        X = np.maximum(0, x)
    elif type == 2:  # LeakyRELU
        X = np.maximum(0.01 * x, x)
    elif type == 3:  # Logistic
        X = 1 / (1 + np.exp(-x))
    elif type == 4:  # tanh
        X = np.tanh(x)
    return X, type
import numpy as np