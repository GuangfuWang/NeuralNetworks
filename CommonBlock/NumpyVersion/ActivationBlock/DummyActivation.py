# Author: Guangfu Wang
# Date: 2022-01-18
# CopyRight Guangfu


class DummyActivation:

    @staticmethod
    def forward(X):
        return X

    @staticmethod
    def backward(dL):
        return dL
