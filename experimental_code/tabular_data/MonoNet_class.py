import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn.functional import softsign
import math
from math import sqrt


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


def init_weights_normal(l):
    if isinstance(l, nn.Linear):
        l.weight.data.normal_(0.0, 1 / sqrt(l.in_features))
        #if l.bias is not None:
            #nn.init.constant_(l.bias.data, 0)


def init_weights_uniform(l):
    if isinstance(l, nn.Linear):
        l.weight.data.uniform_(-1 / sqrt(l.in_features), 1 / sqrt(l.in_features))
        if l.bias is not None:
            nn.init.constant_(l.bias.data, 0)


def init_weights_xavier_uniform(l):
    if isinstance(l, nn.Linear):
        nn.init.xavier_uniform_(l.weight.data)  # , gain=sqrt(l.in_features+l.out_features)/sqrt(6*l.in_features))
        if l.bias is not None:
            nn.init.constant_(l.bias.data, 0)


def init_weights_xavier_normal(l):
    if isinstance(l, nn.Linear):
        nn.init.xavier_normal_(l.weight.data)
        if l.bias is not None:
            nn.init.constant_(l.bias.data, 0)


def init_weights_kaiming_uniform(l):
    if isinstance(l, nn.Linear):
        nn.init.kaiming_uniform_(l.weight.data)
        if l.bias is not None:
            nn.init.constant_(l.bias.data, 0)


def init_weights_kaiming_normal(l):
    if isinstance(l, nn.Linear):
        nn.init.kaiming_normal_(l.weight.data)
        if l.bias is not None:
            nn.init.constant_(l.bias.data, 0)


class MonoNet(nn.Module):
    #    name_layers = ['Input', 'Linear_1', 'Linear_2', 'Interpretable', 'Pre-monotonic', 'Monotonic_1', 'Monotonic_2',
    #      'Output']
    def __init__(self, num_feature, num_class, nb_neuron_inter_layer=8):
        super(MonoNet, self).__init__()

        self.linear_1 = nn.Linear(num_feature, 16)
        # self.linear_1.weight.data.normal_(0.0, 1 / sqrt(self.linear_1.in_features))
        # self.linear_1.weight.data.uniform_(-1 / sqrt(self.linear_1.in_features), 1 / sqrt(self.linear_1.in_features))
        # self.linear_1.weight.data.xavier_uniform_()
        # self.linear_1.weight.data.xavier_normal_()
        # self.linear_1.weight.data.kaiming_uniform_()
        # self.linear_1.weight.data.kaiming_normal_()

        self.linear_2 = nn.Linear(16, 16)
        # self.linear_2.weight.data.normal_(0.0, 1 / sqrt(self.linear_2.in_features))

        self.interpretable = nn.Linear(16, nb_neuron_inter_layer)
        # self.interpretable.weight.data.normal_(0.0, 1 / sqrt(self.interpretable.in_features))

        self.pre_monotonic = InterpretableLayer(nb_neuron_inter_layer, nb_neuron_inter_layer)
        # self.pre_monotonic.weight.data.normal_(0.0, 1 / sqrt(nb_neuron_inter_layer))

        self.monotonic_1 = MonotonicLayer(nb_neuron_inter_layer, 32)
        # self.monotonic_1.weight.data.normal_(0.0, 1 / sqrt(self.monotonic_1.in_features))
        # 1 / sqrt(self.layer_monotonic.in_features)

        self.monotonic_2 = MonotonicLayer(32, num_class)
        # self.monotonic_2.weight.data.normal_(0.0, 1 / sqrt(self.monotonic_2.in_features))
        # 1 / sqrt(self.layer_monotonic_2.in_features)

        self.output = InterpretableLayer(num_class, num_class)
        # self.output.weight.data.normal_(0.0, 1 / sqrt(self.output.in_features))
        # We need to define several activation functions with different names because of a bug with some captum methods
        # self.layer_1.weight.data.uniform_(-1/sqrt(13), 1/sqrt(13))
        # self.layer_2.weight.data.uniform_(-1/sqrt(16), 1/sqrt(16))
        # self.layer_3.weight.data.uniform_(-1/sqrt(16), 1/sqrt(16))
        # self.layer_inter.weight.data.uniform_(-1/sqrt(8), 1/sqrt(8))
        # self.layer_monotonic.weight.data.uniform_(-1/sqrt(8), 1/sqrt(8))
        # self.layer_monotonic_2.weight.data.uniform_(-1/sqrt(16), 1/sqrt(16))
        # self.layer_monotonic_2.weight.data.uniform_(-1/sqrt(16), 1/sqrt(16))
        # self.layer_inter_out.weight.data.uniform_(-1/sqrt(20), 1/sqrt(20))
        act_fct = nn.Tanh()  # nn.LeakyReLU()
        self.activation_fct_lin_1 = act_fct
        # nn.ReLU() nn.Sigmoid() LeakyReLU() Tanh
        self.activation_fct_lin_2 = act_fct
        self.activation_fct_inter = act_fct
        self.activation_fct_mon_1 = act_fct
        self.activation_fct_mon_2 = act_fct

        self.accuracy_train = None
        self.accuracy_val = None
        self.accuracy_test = None
        self.accuracy_hist = None
        self.loss_hist = None

    def forward(self, x):
        x = self.unconstrained_block(x)
        x = self.monotonic_block(x)

        return x

    def unconstrained_block(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)
        x = self.linear_1(x)
        x = self.activation_fct_lin_1(x)

        x = self.linear_2(x)
        x = self.activation_fct_lin_2(x)

        x = self.interpretable(x)
        x = self.activation_fct_inter(x)

        return x

    def unconstrained_block_to_numpy(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.unconstrained_block(x)

        return x.detach().numpy()

    def monotonic_block(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.pre_monotonic(x)
        x = nn.functional.softsign(x)

        x = self.monotonic_1(x)
        x = self.activation_fct_mon_1(x)

        x = self.monotonic_2(x)
        x = self.activation_fct_mon_2(x)

        x = self.output(x)
        x = nn.functional.softsign(x)

        return x

    def get_linear_1(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)
        x = self.linear_1(x)
        x = self.activation_fct_lin_1(x)

        return x

    def get_linear_2(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)
        x = self.linear_1(x)
        x = self.activation_fct_lin_1(x)

        x = self.linear_2(x)
        x = self.activation_fct_lin_2(x)

        return x

    def get_interpretable(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)
        x = self.unconstrained_block(x)

        return x

    def get_pre_monotonic(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)
        x = self.unconstrained_block(x)
        x = self.pre_monotonic(x)
        x = nn.functional.softsign(x)

        return x

    def get_monotonic_1(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)
        x = self.unconstrained_block(x)

        x = self.pre_monotonic(x)
        x = nn.functional.softsign(x)

        x = self.monotonic_1(x)
        x = self.activation_fct_mon_1(x)

        return x

    def get_monotonic_2(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)
        x = self.unconstrained_block(x)

        x = self.pre_monotonic(x)
        x = nn.functional.softsign(x)

        x = self.monotonic_1(x)
        x = self.activation_fct_mon_1(x)

        x = self.monotonic_2(x)
        x = self.activation_fct_mon_2(x)

        return x

    def last_layer(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.unconstrained_block(x)

        x = self.pre_monotonic(x)
        x = nn.functional.softsign(x)

        x = self.monotonic_1(x)
        x = self.activation_fct_mon_1(x)

        x = self.monotonic_2(x)
        x = self.activation_fct_mon_2(x)

        x = self.output(x)
        x = nn.functional.softsign(x)

        return x

    def unconstrained_block_0(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.unconstrained_block(x)

        return x[:, 0].reshape(len(x), 1).detach().numpy()

    def unconstrained_block_0_torch(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.unconstrained_block(x)

        return x[:, 0].reshape(len(x), 1)

    def unconstrained_block_1(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.unconstrained_block(x)

        return x[:, 1].reshape(len(x), 1).detach().numpy()

    def unconstrained_block_1_torch(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.unconstrained_block(x)

        return x[:, 1].reshape(len(x), 1)

    def unconstrained_block_2(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.unconstrained_block(x)

        return x[:, 2].reshape(len(x), 1).detach().numpy()

    def unconstrained_block_2_torch(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.unconstrained_block(x)

        return x[:, 2].reshape(len(x), 1)

    def unconstrained_block_3(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.unconstrained_block(x)

        return x[:, 3].reshape(len(x), 1).detach().numpy()

    def unconstrained_block_3_torch(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.unconstrained_block(x)

        return x[:, 3].reshape(len(x), 1)

    def unconstrained_block_4(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.unconstrained_block(x)

        return x[:, 4].reshape(len(x), 1).detach().numpy()

    def unconstrained_block_4_torch(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.unconstrained_block(x)

        return x[:, 4].reshape(len(x), 1)

    def unconstrained_block_5(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.unconstrained_block(x)

        return x[:, 5].reshape(len(x), 1).detach().numpy()

    def unconstrained_block_5_torch(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.unconstrained_block(x)

        return x[:, 5].reshape(len(x), 1)

    def unconstrained_block_6(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.unconstrained_block(x)

        return x[:, 6].reshape(len(x), 1).detach().numpy()

    def unconstrained_block_6_torch(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.unconstrained_block(x)

        return x[:, 6].reshape(len(x), 1)

    def unconstrained_block_7(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.unconstrained_block(x)

        return x[:, 7].reshape(len(x), 1).detach().numpy()

    def unconstrained_block_7_torch(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.unconstrained_block(x)

        return x[:, 7].reshape(len(x), 1)
