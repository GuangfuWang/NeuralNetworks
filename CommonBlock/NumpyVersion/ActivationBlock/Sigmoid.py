# Author: Guangfu Wang
# Date: 2022-01-17
# CopyRight Guangfu


import numpy as np


class NumpySigmoid:
    numpy_sigmoid = None

    @staticmethod
    def forward(X):
        NumpySigmoid.numpy_sigmoid = 1 / np.exp(-X)
        return NumpySigmoid.numpy_sigmoid

    @staticmethod
    def backward(dL):
        return NumpySigmoid.numpy_sigmoid * (1 - NumpySigmoid.numpy_sigmoid) * dL
