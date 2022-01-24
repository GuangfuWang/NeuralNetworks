# Author: Guangfu Wang
# Date: 2022-01-17
# CopyRight Guangfu

import numpy as np
from CommonBlock.NumpyVersion.ActivationBlock.ActivationFactory import ActivationFactory


class FullyConnectedBlock:
    def __init__(self, in_dim: int, out_dim: int, activation: str = 'None'):
        assert in_dim >= 0, "You must give a valid in_dim"
        assert out_dim >= 0, "You must give a valid out_dim"
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weights = np.random.standard_normal((out_dim, in_dim))
        self.bias = np.random.standard_normal(out_dim)
        self.weights_gradient = np.full_like(self.weights, 0.0)
        self.bias_gradient = np.full_like(self.bias, 0.0)
        self.activation = FullyConnectedBlock.__get_activation_func__(activation)
        self.num_params = 2 * out_dim * (in_dim + 1)
        self.input = None
        self.output = None

    def forward(self, inputs):
        """Note here inputs can be 1-dim or 2-dim row-major numpy array, with rows represent each sample."""
        biased = np.einsum('ij,kj->ki', self.weights, inputs) + self.bias
        self.input = inputs
        self.output = biased
        return self.activation.forward(biased)

    def backward(self, lr: float = 0.0001, weight_decay: float = 0.00004):
        self.weights *= (1 - weight_decay)
        self.bias *= (1 - weight_decay)
        self.weights -= lr * self.weights_gradient
        self.bias -= lr * self.bias_gradient
        """Empty gradients"""
        self.weights_gradient = np.zeros(self.weights.shape)
        self.bias_gradient = np.zeros(self.bias.shape)

    def gradient(self, out_gradient):
        ndim = 1
        if self.input.ndim > 1:
            ndim = self.input.shape[0]
        local_gradient = self.activation.backward(out_gradient)
        local_prev_gradient = np.matmul(local_gradient, self.weights)
        self.weights_gradient = np.einsum('ij,ik->jk', local_gradient, self.input) * (1 / ndim)
        self.bias_gradient = np.einsum('ij->j', local_gradient) * (1 / ndim)
        return local_prev_gradient

    @staticmethod
    def __get_activation_func__(activation: str = 'None'):
        """Default activation function is relu."""
        """If none is specified, no activation will be performed."""
        return ActivationFactory.createActivation(activation.lower())
