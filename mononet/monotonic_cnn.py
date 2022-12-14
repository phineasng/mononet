import torch

from .mononet import NNBaseClass, MonotonicLayer, InterpretableLayer, MonotonicConv2d
from torch import nn


class ResidualMonotonicCNN1(NNBaseClass):
    def __init__(self, nclasses, shape=(28, 28), nchannels=3, lr=0.001, optimizer='sgd', with_monotonic=True):
        super(ResidualMonotonicCNN1, self).__init__(lr=lr, optimizer=optimizer, n_class=nclasses)
        self._shape = shape
        self._nchannels = nchannels
        self._nclasses = nclasses
        if not hasattr(self, 'topk'):
            self.topk = 5
        if not hasattr(self, 'n_maps'):
            self.n_maps = 256
        if not hasattr(self, 'monotonic_pos_fn'):
            self.monotonic_pos_fn = 'sigmoid'
        self.with_monotonic = with_monotonic

        self.feat1 = nn.Conv2d(self._nchannels, self.n_maps, kernel_size=5, padding=2)
        self.interpret = InterpretableLayer(self.topk*self.n_maps)
        self.monotonic_block = nn.Sequential(
            MonotonicLayer(self.topk*self.n_maps, 64, fn=self.monotonic_pos_fn),
            nn.LeakyReLU(),
            MonotonicLayer(64, nclasses, fn=self.monotonic_pos_fn),
            nn.LeakyReLU()
        )
        self.interpret_out = InterpretableLayer(nclasses)
        self.residual = nn.Sequential(
            MonotonicLayer(self.topk * self.n_maps, nclasses, fn='tanh_p1'),
            nn.LeakyReLU()
        )
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.leaky = nn.LeakyReLU()
        self.save_hyperparameters()

    def get_feature_maps(self, x):
        x = self.feat1(x)
        x = self.tanh(x)
        return x

    def get_features(self, x):
        n = len(x)
        feat_maps = self.get_feature_maps(x)
        x, pos = torch.topk(torch.reshape(feat_maps, (n, self.n_maps, -1)), k=self.topk, dim=-1)
        x = x.reshape(n, self.topk*self.n_maps)
        return x, pos

    def forward(self, x):
        x, pos = self.get_features(x)
        res = self.residual(x)
        if self.with_monotonic:
            x = self.interpret(x)
            x = self.monotonic_block(x)
            x = (self.interpret_out(x) + res)*.5
        else:
            x = res
        return x


class ResidualMonotonicCNN2(ResidualMonotonicCNN1):
    def __init__(self, nclasses, shape=(28, 28), nchannels=3, lr=0.001, optimizer='sgd', with_monotonic=True):
        self.topk = 64
        self.n_maps = 256
        self.monotonic_pos_fn = 'tanh_p1'
        super(ResidualMonotonicCNN2, self).__init__(nclasses, shape=shape, nchannels=nchannels,
                                                    lr=lr, optimizer=optimizer, with_monotonic=with_monotonic)


class ResidualMonotonicCNN3(ResidualMonotonicCNN1):
    def __init__(self, nclasses, shape=(28, 28), nchannels=3, lr=0.001, optimizer='sgd', with_monotonic=True):
        self.topk = 32
        self.n_maps = 512
        self.monotonic_pos_fn = 'tanh_p1'
        super(ResidualMonotonicCNN3, self).__init__(nclasses, shape=shape, nchannels=nchannels,
                                                    lr=lr, optimizer=optimizer, with_monotonic=with_monotonic)
        self.feat1 = nn.Sequential(
            nn.Conv2d(self._nchannels, self.n_maps, kernel_size=9, padding=4),
            nn.Dropout2d(p=0.2)
        )


class ResidualMonotonicCNN4(ResidualMonotonicCNN1):
    def __init__(self, nclasses, shape=(28, 28), nchannels=3, lr=0.001, optimizer='sgd', with_monotonic=True):
        self.topk = 8
        self.n_maps = 128
        self.monotonic_pos_fn = 'tanh_p1'
        super(ResidualMonotonicCNN4, self).__init__(nclasses, shape=shape, nchannels=nchannels,
                                                    lr=lr, optimizer=optimizer, with_monotonic=with_monotonic)
        self.residual = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.topk*self.n_maps, nclasses)
        )


class ResidualMonotonicCNN5(ResidualMonotonicCNN1):
    def __init__(self, nclasses, shape=(28, 28), nchannels=3, lr=0.001, optimizer='sgd', with_monotonic=True):
        self.topk = 16
        self.n_maps = 128
        self.monotonic_pos_fn = 'sigmoid'
        super(ResidualMonotonicCNN5, self).__init__(nclasses, shape=shape, nchannels=nchannels,
                                                    lr=lr, optimizer=optimizer, with_monotonic=with_monotonic)
        self.residual = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.topk*self.n_maps, nclasses, bias=False)
        )


class ResidualMonotonicCNN6(ResidualMonotonicCNN1):
    def __init__(self, nclasses, shape=(28, 28), nchannels=3, lr=0.001, optimizer='sgd', with_monotonic=True):
        self.topk = 16
        self.n_maps = 128
        self.monotonic_pos_fn = 'sigmoid'
        super(ResidualMonotonicCNN6, self).__init__(nclasses, shape=shape, nchannels=nchannels,
                                                    lr=lr, optimizer=optimizer, with_monotonic=with_monotonic)
        self.feat1 = nn.Sequential(
            nn.Conv2d(self._nchannels, 64, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            MonotonicConv2d(64, self.n_maps, kernel_size=9, padding=4),
            nn.ReLU()
        )


class ResidualMonotonicCNN7(ResidualMonotonicCNN1):
    def forward(self, x):
        x, pos = self.get_features(x)
        x = self.interpret(x)
        res = self.residual(x)
        if self.with_monotonic:
            x = self.monotonic_block(x)
            x = self.interpret_out((x + res)*.5)
        else:
            x = res
        return x