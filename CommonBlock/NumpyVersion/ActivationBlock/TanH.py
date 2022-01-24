# Author: Guangfu Wang
# Date: 2022-01-18
# CopyRight Guangfu

import numpy as np


class NumpyTanH:
    positive = None
    negative = None

    @staticmethod
    def forward(X):
        NumpyTanH.positive = np.exp(X)
        NumpyTanH.negative = np.exp(-X)
        return (NumpyTanH.positive - NumpyTanH.negative) / (NumpyTanH.positive + NumpyTanH.negative)

    @staticmethod
    def backward(dL):
        return dL * (4.0 / (NumpyTanH.negative + NumpyTanH.positive) ** 2)
