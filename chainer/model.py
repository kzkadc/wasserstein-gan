# coding: utf-8

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, iterators, Chain, optimizers
from chainer.training import updaters, Trainer, extensions


def compose(x, funcs):
    y = x
    for f in funcs:
        y = f(y)

    return y


class Critic(Chain):
    def __init__(self):
        super().__init__()
        kwds = {
            "ksize": 4,
            "stride": 2,
            "pad": 1
        }
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 32, **kwds)		# (14,14)
            self.conv2 = L.Convolution2D(32, 64, **kwds)		# (7,7)
            self.conv3 = L.Convolution2D(64, 128, ksize=2, stride=1, pad=0)		# (6,6)
            self.conv4 = L.Convolution2D(128, 256, **kwds)  # (3,3)
            self.conv5 = L.Convolution2D(256, 1, ksize=1, stride=1, pad=0)

    def __call__(self, x):
        h = compose(x, [
            self.conv1, F.leaky_relu,
            self.conv2, F.leaky_relu,
            self.conv3, F.leaky_relu,
            self.conv4, F.leaky_relu,
            self.conv5,
            lambda x:F.mean(x, axis=(1, 2, 3))  # global average pooling
        ])

        return h


class Generator(Chain):
    def __init__(self, z_dim):
        super().__init__()
        with self.init_scope():
            self.fc = L.Linear(z_dim, 3 * 3 * 256)
            self.bn0 = L.BatchNormalization(3 * 3 * 256)
            self.dc1 = L.Deconvolution2D(256, 128, ksize=4, stride=2, pad=1, nobias=True)
            self.bn1 = L.BatchNormalization(128)
            self.dc2 = L.Deconvolution2D(128, 64, ksize=2, stride=1, pad=0, nobias=True)
            self.bn2 = L.BatchNormalization(64)
            self.dc3 = L.Deconvolution2D(64, 32, ksize=4, stride=2, pad=1, nobias=True)
            self.bn3 = L.BatchNormalization(32)
            self.dc4 = L.Deconvolution2D(32, 1, ksize=4, stride=2, pad=1)

    def __call__(self, z):
        h = compose(z, [
            self.fc, self.bn0, F.leaky_relu,
            lambda x:F.reshape(x, (-1, 256, 3, 3)),
            self.dc1, self.bn1, F.leaky_relu,
            self.dc2, self.bn2, F.leaky_relu,
            self.dc3, self.bn3, F.leaky_relu,
            self.dc4, F.sigmoid
        ])

        return h
