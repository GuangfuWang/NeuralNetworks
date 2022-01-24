# Author: Guangfu Wang
# Date: 2022-01-17
# CopyRight Guangfu

from functools import reduce
import numpy as np
import math


class NumpyConvBlock:
    def __init__(self, in_shape: tuple, out_channel: int,
                 kernel_size_h: int = 3, kernel_size_w: int = 3,
                 padding: str = 'valid', stride: int = 1):
        self.in_shape = in_shape
        self.out_channel = out_channel
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        padding = padding.lower()
        self.padding = padding
        self.stride = stride

        """Parameters need to update."""
        weights_scale = math.sqrt(reduce(lambda x, y: x * y, in_shape) / out_channel)
        self.kernels = np.random.standard_normal((out_channel,
                                                  kernel_size_h,
                                                  kernel_size_w,
                                                  in_shape[-1])) / weights_scale
        self.bias = np.random.standard_normal(out_channel) / weights_scale

        """Gradients."""
        self.kernels_gradient = np.full_like(self.kernels, 0.0, dtype=np.float)
        self.bias_gradient = np.full_like(self.bias, 0.0, dtype=np.float)

        self.out_shape = NumpyConvBlock.__calculate_out_shape__(in_shape,
                                                                padding,
                                                                out_channel,
                                                                kernel_size_h,
                                                                kernel_size_w,
                                                                stride)
        self.out_gradient = np.full(self.out_shape, 0.0, dtype=np.float)
        self.col_img = None

    def forward(self, inputs):
        """first padding images"""
        if self.padding is 'same':
            p_h = self.kernel_size_h / 2
            p_w = self.kernel_size_w / 2
            inputs = np.pad(inputs,
                            ((0, 0),
                             (p_h, p_h),
                             (p_w, p_w),
                             (0, 0)),
                            'constant', constant_values=0)
        """convert images to vectors for fast multiplication"""
        self.col_img = NumpyConvBlock.__transform_to_column__(inputs, self.in_shape, self.out_shape, self.kernel_size_h,
                                                              self.kernel_size_w, self.stride)
        """perform convolution operations"""
        transformed_kernel = self.kernels.reshape(
            (-1, self.kernel_size_h * self.kernel_size_w * self.in_shape[-1])).transpose([1, 0])
        conv = np.dot(self.col_img, transformed_kernel)
        conv = np.einsum('ij,j->ij', conv, self.bias).reshape(self.out_shape)
        return conv

    def backward(self, lr: float = 0.0005, weight_decay: float = 0.00004):
        """update all gradient"""
        self.kernels *= (1 - weight_decay)
        self.bias *= (1 - weight_decay)
        self.kernels -= lr * self.kernels_gradient
        self.bias -= lr * self.bias_gradient

        self.kernels_gradient = self.bias_gradient.fill(0.0)
        self.bias_gradient = self.bias_gradient.fill(0.0)

    def gradient(self, out_gradients):
        """first we need to make sure our gradient compatible with out-shape."""
        assert out_gradients.shape == self.out_shape, "Your out-gradient's shape not compatible with out-shape."
        """compute kernel gradient and bias gradient: delta loss/delta kernel_parameters && bias"""
        transformed_out_gradient = out_gradients.reshape((-1, self.out_channel))
        for i in range(self.in_shape[0]):
            t = self.in_shape[1] * self.in_shape[2]
            self.kernels_gradient += np.einsum('ij,jk->ik', self.col_img[i * t::t],
                                               transformed_out_gradient).transpose([1, 0]).reshape(self.kernels.shape)
        self.bias_gradient += np.sum(transformed_out_gradient, axis=0)
        """compute gradient for previous layer: delta loss/delta inputs"""
        kernels = np.flipud(np.fliplr(self.kernels.transpose([1, 2, 0, 3]))).reshape((-1, self.in_shape[-1]))
        """padding for de-convolution operation"""
        pad_eta = None
        if self.padding == 'valid':
            pad_eta = np.pad(out_gradients, (
                (0, 0), (self.kernel_size_h - 1, self.kernel_size_h - 1),
                (self.kernel_size_w - 1, self.kernel_size_w - 1), (0, 0)),
                             'constant', constant_values=0)

        if self.padding == 'same':
            pad_eta = np.pad(out_gradients, (
                (0, 0), (self.kernel_size_h / 2, self.kernel_size_h / 2),
                (self.kernel_size_w / 2, self.kernel_size_w / 2), (0, 0)),
                             'constant', constant_values=0)

        col_pad_eta = np.array(
            [NumpyConvBlock.__transform_to_column__(pad_eta[i][np.newaxis, :], self.in_shape, self.out_shape,
                                                    self.kernel_size_h, self.kernel_size_w, self.stride)
             for i in range(self.in_shape[0])])
        prev_gradient = np.dot(col_pad_eta, kernels)
        prev_gradient = np.reshape(prev_gradient, self.in_shape)
        """return previous gradient"""
        return prev_gradient

    @staticmethod
    def __transform_to_column__(images, in_shape, out_shape, kernel_size_h, kernel_size_w, stride):
        assert images.shape == in_shape, "Your defined input shape is not compatible with your actual input!"
        batch, h, w, c = images.shape
        col_img = np.empty(
            (batch * out_shape[1] * out_shape[2], kernel_size_h * kernel_size_w * c))
        outsize = out_shape[1] * out_shape[2]

        for y in range(out_shape[1]):
            y_min = y * stride
            y_max = y_min + kernel_size_h
            y_start = y * out_shape[2]
            for x in range(out_shape[2]):
                x_min = x * stride
                x_max = x_min + kernel_size_w
                col_img[y_start + x::outsize, :] = images[:, y_min:y_max, x_min:x_max, :].reshape(batch, -1)
        return col_img

    @staticmethod
    def __calculate_out_shape__(in_shape: tuple, padding: str, out_channel: int, ksize_h: int, ksize_w: int,
                                stride: int):
        batch_size, h, w, in_channel = in_shape
        out_shape = (batch_size, -1, -1, -1)
        if padding is 'valid':
            o_h = (h - ksize_h + 1) / stride
            o_w = (w - ksize_w + 1) / stride
            out_shape = (batch_size, o_h, o_w, out_channel)
        elif padding is 'same':
            o_h = h / stride
            o_w = w / stride
            out_shape = (batch_size, o_h, o_w, out_channel)
        else:
            raise ValueError("You specify an unsupported padding mode, please use valid/same.")

        return out_shape
