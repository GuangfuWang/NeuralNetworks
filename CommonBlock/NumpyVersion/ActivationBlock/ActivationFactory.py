# Author: Guangfu Wang
# Date: 2022-01-20
# CopyRight Guangfu

from CommonBlock.NumpyVersion.ActivationBlock import \
    DummyActivation, GELU, Sigmoid, SoftMax, TanH, ReLu


class ActivationFactory:

    @staticmethod
    def createActivation(activation: str = 'relu'):
        if activation is 'relu':
            return ReLu.NumpyReLU
        elif activation is 'gelu':
            return GELU.NumpyVanillaGELU
        elif activation is 'sigmoid':
            return Sigmoid.NumpySigmoid
        elif activation is 'softmax':
            return SoftMax.NumpySoftmax
        elif activation is 'tanh':
            return TanH.NumpyTanH
        elif activation is 'none':
            return DummyActivation.DummyActivation
        else:
            return ReLu.NumpyReLU
