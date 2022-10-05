import torch

from .mononet import NNBaseClass, MonotonicLayer, InterpretableLayer
from torch import nn


class SingleCellMonoNetExample(NNBaseClass):
    def __init__(self, **kwargs):
        super(SingleCellMonoNetExample, self).__init__(**kwargs)

        self.high_level_feats = nn.Sequential(
            nn.Linear(13, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 8),
            nn.LeakyReLU()
        )

        self.pre_monotonic = InterpretableLayer(8)
        # self.pre_monotonic.weight.data.normal_(0.0, 1 / sqrt(nb_neuron_inter_layer))

        self.monotonic = nn.Sequential(
            MonotonicLayer(8, 32, fn='tanh_p1'),
            nn.LeakyReLU(),
            MonotonicLayer(32, 16, fn='tanh_p1'),
            nn.LeakyReLU(),
            MonotonicLayer(16, 20, fn='tanh_p1'),
        )

        self.output = InterpretableLayer(20)

    def forward(self, x):
        x = self.high_level_feats(x)
        x = self.pre_monotonic(x)
        x = self.monotonic(x)
        x = self.output(x)
        return x
