# Author: Guangfu Wang
# Date: 2022-01-17
# CopyRight Guangfu


import numpy as np


class NumpyReLU:
    input = None

    @staticmethod
    def forward(X):
        NumpyReLU.input = X
        return np.max(0, X)

    @staticmethod
    def backward(dL):
        dX = np.copy(dL)
        dX[NumpyReLU.input <= 0] = 0
        return dX
