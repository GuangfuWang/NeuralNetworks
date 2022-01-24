# Author: Guangfu Wang
# Date: 2022-01-17
# CopyRight Guangfu
import numpy as np


class NumpyVanillaGELU:
    positive = None
    negative = None
    two_over_pi = np.sqrt(2.0 / np.pi)
    inter_var = 0.0
    tanh_ed = None
    input = None

    @staticmethod
    def forward(X):
        NumpyVanillaGELU.input = X
        NumpyVanillaGELU.inter_var = NumpyVanillaGELU.two_over_pi * (X + 0.044715 * X ** 3)
        NumpyVanillaGELU.positive = np.exp(NumpyVanillaGELU.inter_var)
        NumpyVanillaGELU.negative = np.exp(-NumpyVanillaGELU.inter_var)
        NumpyVanillaGELU.tanh_ed = (NumpyVanillaGELU.positive - NumpyVanillaGELU.negative) / (
                NumpyVanillaGELU.positive + NumpyVanillaGELU.negative)
        return 0.5 * X * (1 + NumpyVanillaGELU.tanh_ed)

    @staticmethod
    def backward(dL):
        first = 0.5 * (1 + NumpyVanillaGELU.tanh_ed)
        second = 0.5 * NumpyVanillaGELU.input * \
                 (NumpyVanillaGELU.two_over_pi + 0.044715 * 3 * NumpyVanillaGELU.input ** 2) * \
                 (4.0 / (NumpyVanillaGELU.negative + NumpyVanillaGELU.positive) ** 2)
        return dL * (first + second)
