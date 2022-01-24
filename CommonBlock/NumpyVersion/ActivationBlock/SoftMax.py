# Author: Guangfu Wang
# Date: 2022-01-17
# CopyRight Guangfu

import numpy as np


class NumpySoftmax:
    denominator = 1.0
    softmax = None

    @staticmethod
    def forward(X):
        """Input X can be any-array-like objects."""
        """We use multiply here because multiplication is much faster than division."""
        NumpySoftmax.denominator = 1 / np.sum(np.exp(X))
        NumpySoftmax.softmax = np.exp(X) * NumpySoftmax.denominator
        return NumpySoftmax.softmax

    @staticmethod
    def backward(dL):
        local_gradient = -np.einsum('ki,kj->ijk', NumpySoftmax.softmax, NumpySoftmax.softmax)
        return np.matmul(dL, local_gradient) + dL * NumpySoftmax.softmax
