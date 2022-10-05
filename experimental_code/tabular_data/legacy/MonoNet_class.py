import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn.functional import softsign
import math


class InterpretableLayer(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch

    def __init__(self, in_features: int, out_features: int) -> None:
        super(InterpretableLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features))
        #  self.reset_parameters()

    def forward(self, input: torch) -> torch:
        #  return input*torch.exp(self.weight) + self.bias  # DONE: take exp away an bias and add softsign
        return input * self.weight


class MonotonicLayer(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(MonotonicLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch) -> torch:
        return torch.matmul(input, torch.transpose(torch.exp(self.weight), 0, 1)) + self.bias


class MonoNet(nn.Module):
    def __init__(self, num_feature, num_class, nb_neuron_inter_layer=8):
        super(MonoNet, self).__init__()

        self.layer_1 = nn.Linear(num_feature, 64)
        self.layer_2 = nn.Linear(64, 64)
        # self.layer_3 = nn.Linear(64, 32)
        self.layer_4 = nn.Linear(64, nb_neuron_inter_layer)

        self.layer_inter = InterpretableLayer(nb_neuron_inter_layer, nb_neuron_inter_layer)
        self.layer_inter.weight.data.uniform_(-0.15, 0.15)
        self.layer_monotonic = MonotonicLayer(nb_neuron_inter_layer, 64)
        self.layer_monotonic_2 = MonotonicLayer(64, 64)
        self.layer_out = MonotonicLayer(64, num_class)  # Why did I put a linear layer here? And not a monotonic layer ?

        # We need to define several activation functions with different names because of a bug somewhere
        self.activation_fct = nn.Tanh()
        self.activation_fct_1 = nn.Tanh()  # nn.ReLU() nn.Sigmoid()
        self.activation_fct_2 = nn.Tanh()
        self.activation_fct_3 = nn.Tanh()
        self.activation_fct_4 = nn.Tanh()
        self.activation_fct_5 = nn.Tanh()
        self.accuracy_train = None
        self.accuracy_val = None
        self.accuracy_test = None

    def forward(self, x):
        # DONE: try to remove batchnorm and dropout
        x = self.layer_1(x)
        x = self.activation_fct_1(x)

        x = self.layer_2(x)
        x = self.activation_fct_2(x)

        # x = self.layer_3(x)
        # x = self.activation_fct_3(x)

        x = self.layer_4(x)
        x = self.activation_fct_3(x)

        x = self.layer_inter(x)
        x = nn.functional.softsign(x)

        x = self.layer_monotonic(x)
        x = self.activation_fct_4(x)

        x = self.layer_monotonic_2(x)
        x = self.activation_fct_5(x)

        x = self.layer_out(x)
        x = nn.functional.softsign(x)

        return x

    def unconstrainted_block(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)
        x = self.layer_1(x)
        x = self.activation_fct_1(x)

        x = self.layer_2(x)
        x = self.activation_fct_2(x)

        # x = self.layer_3(x)
        # x = self.activation_fct_3(x)

        x = self.layer_4(x)
        x = self.activation_fct_3(x)

        return x

    def unconstrainted_block_to_numpy(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.layer_1(x)
        x = self.activation_fct_1(x)

        x = self.layer_2(x)
        x = self.activation_fct_2(x)

        # x = self.layer_3(x)
        # x = self.activation_fct_3(x)

        x = self.layer_4(x)
        x = self.activation_fct_3(x)

        return x.detach().numpy()

    def monotonic_block(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.layer_inter(x)
        x = nn.functional.softsign(x)

        x = self.layer_monotonic(x)
        x = self.activation_fct_4(x)

        x = self.layer_monotonic_2(x)
        x = self.activation_fct_5(x)

        x = self.layer_out(x)
        x = nn.functional.softsign(x)

        return x

    def unconstrainted_block_0(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)
        x = self.layer_1(x)
        x = self.activation_fct_1(x)

        x = self.layer_2(x)
        x = self.activation_fct_2(x)

        # x = self.layer_3(x)
        # x = self.activation_fct_3(x)

        x = self.layer_4(x)
        x = self.activation_fct_3(x)

        return x[:, 0].detach().numpy()

    def unconstrainted_block_1(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)
        x = self.layer_1(x)
        x = self.activation_fct_1(x)

        x = self.layer_2(x)
        x = self.activation_fct_2(x)

        # x = self.layer_3(x)
        # x = self.activation_fct_3(x)

        x = self.layer_4(x)
        x = self.activation_fct_3(x)

        return x[:, 1].detach().numpy()

    def unconstrainted_block_2(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)
        x = self.layer_1(x)
        x = self.activation_fct_1(x)

        x = self.layer_2(x)
        x = self.activation_fct_2(x)

        # x = self.layer_3(x)
        # x = self.activation_fct_3(x)

        x = self.layer_4(x)
        x = self.activation_fct_3(x)

        return x[:, 2].detach().numpy()

    def unconstrainted_block_3(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)
        x = self.layer_1(x)
        x = self.activation_fct_1(x)

        x = self.layer_2(x)
        x = self.activation_fct_2(x)

        # x = self.layer_3(x)
        # x = self.activation_fct_3(x)

        x = self.layer_4(x)
        x = self.activation_fct_3(x)

        return x[:, 3].detach().numpy()

    def unconstrainted_block_4(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)
        x = self.layer_1(x)
        x = self.activation_fct_1(x)

        x = self.layer_2(x)
        x = self.activation_fct_2(x)

        # x = self.layer_3(x)
        # x = self.activation_fct_3(x)

        x = self.layer_4(x)
        x = self.activation_fct_3(x)

        return x[:, 4].detach().numpy()

    def unconstrainted_block_5(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)
        x = self.layer_1(x)
        x = self.activation_fct_1(x)

        x = self.layer_2(x)
        x = self.activation_fct_2(x)

        # x = self.layer_3(x)
        # x = self.activation_fct_3(x)

        x = self.layer_4(x)
        x = self.activation_fct_3(x)

        return x[:, 5].detach().numpy()

    def unconstrainted_block_6(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)
        x = self.layer_1(x)
        x = self.activation_fct_1(x)

        x = self.layer_2(x)
        x = self.activation_fct_2(x)

        # x = self.layer_3(x)
        # x = self.activation_fct_3(x)

        x = self.layer_4(x)
        x = self.activation_fct_3(x)

        return x[:, 6].detach().numpy()

    def unconstrainted_block_7(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)
        x = self.layer_1(x)
        x = self.activation_fct_1(x)

        x = self.layer_2(x)
        x = self.activation_fct_2(x)

        # x = self.layer_3(x)
        # x = self.activation_fct_3(x)

        x = self.layer_4(x)
        x = self.activation_fct_3(x)

        return x[:, 7].detach().numpy()
