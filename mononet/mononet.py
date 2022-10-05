import torch
import torch.nn as nn
import torchmetrics
from torch.nn.parameter import Parameter
from torch.nn import init
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim import Adam, SGD
import math


class InterpretableLayer(nn.Module):
    __constants__ = ['in_features']
    in_features: int
    out_features: int
    weight: torch

    def __init__(self, in_features: int) -> None:
        super(InterpretableLayer, self).__init__()
        self.in_features = in_features
        self.weight = Parameter(torch.Tensor(in_features))
        self.softsign = nn.Softsign()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.normal_(self.weight, mean=0)

    def forward(self, input: torch) -> torch:
        #  return input*torch.exp(self.weight) + self.bias  # DONE: take exp away an bias and add softsign
        return input * self.weight


class MonotonicLayer(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch

    def pos_tanh(self, x):
        return torch.tanh(x) + 1.

    def __init__(self, in_features: int, out_features: int, bias: bool = True, fn: str = 'exp') -> None:
        super(MonotonicLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.fn = fn
        if fn == 'exp':
            self.pos_fn = torch.exp
        elif fn == 'square':
            self.pos_fn = torch.square
        elif fn == 'abs':
            self.pos_fn = torch.abs
        elif fn == 'sigmoid':
            self.pos_fn = torch.sigmoid
        else:
            self.fn = 'tanh_p1'
            self.pos_fn = self.pos_tanh
        self.reset_parameters()

    def reset_parameters(self) -> None:
        n_in = self.in_features
        if self.fn == 'exp':
            mean = math.log(1./n_in)
        else:
            mean = 0
        init.normal_(self.weight, mean=mean)
        if self.bias is not None:
            init.uniform_(self.bias, -1./n_in, 1./n_in)

    def forward(self, input: torch) -> torch:
        ret = torch.matmul(input, torch.transpose(self.pos_fn(self.weight), 0, 1))
        if self.bias is not None:
            ret = ret + self.bias
        return ret


class MonotonicConv2d(nn.Module):
    def pos_tanh(self, x):
        return torch.tanh(x) + 1.

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, fn='tanh_p1'):
        super(MonotonicConv2d, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        self._filters = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        self.fn = fn
        if fn == 'exp':
            self.pos_fn = torch.exp
        elif fn == 'square':
            self.pos_fn = torch.square
        elif fn == 'abs':
            self.pos_fn = torch.abs
        elif fn == 'sigmoid':
            self.pos_fn = torch.sigmoid
        else:
            self.fn = 'tanh_p1'
            self.pos_fn = self.pos_tanh
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x):
        filters = self.pos_fn(self._filters)
        return F.conv2d(x, filters, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation,
                        groups=self.groups)


class NNBaseClass(LightningModule):
    def __init__(self, lr: float = 0.001, optimizer='adam'):
        super().__init__()
        self._lr = lr
        self._optimizer = optimizer
        self.accuracy = torchmetrics.Accuracy()

    def configure_optimizers(self):
        if self._optimizer == 'adam':
            optimizer = Adam(params=self.parameters(), lr=self._lr, betas=(0.9, 0.95))
        elif self._optimizer == 'sgd':
            optimizer = SGD(params=self.parameters(), lr=self._lr)
        else:
            raise RuntimeError
        return optimizer

    def _step(self, batch, batch_idx, step_name):
        x, y = batch
        y = torch.flatten(y)
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        self.log(f'{step_name}/accuracy', accuracy, prog_bar=True)
        return {
            'loss': loss
        }

    def training_step(self, batch, batch_idx):
        self.train()
        return self._step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        self.eval()
        ret = self._step(batch, batch_idx, 'valid')
        self.train()
        return ret

    def test_step(self, batch, batch_idx):
        self.eval()
        ret = self._step(batch, batch_idx, 'test')
        self.train()
        return ret

