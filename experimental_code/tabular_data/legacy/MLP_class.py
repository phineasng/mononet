import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn.functional import softsign
import math
from math import sqrt


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
        nn.init.xavier_uniform_(l.weight.data)
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


class MLP(nn.Module):
    #    name_layers = ['Input', 'Linear_1', 'Linear_2', 'Interpretable', 'Pre-monotonic', 'Monotonic_1', 'Monotonic_2',
    #      'Output']
    def __init__(self, num_feature, num_class, nb_neuron_inter_layer=8):
        super(MLP, self).__init__()

        self.linear_1 = nn.Linear(num_feature, 16)

        self.linear_2 = nn.Linear(16, 16)
        # self.linear_2.weight.data.normal_(0.0, 1 / sqrt(self.linear_2.in_features))

        self.output = nn.Linear(16, num_class)
        # self.interpretable.weight.data.normal_(0.0, 1 / sqrt(self.interpretable.in_features))
        act_fct = nn.Tanh()  # nn.LeakyReLU()
        self.activation_fct_lin_1 = act_fct
        # nn.ReLU() nn.Sigmoid() LeakyReLU() Tanh
        self.activation_fct_lin_2 = act_fct

        self.accuracy_train = None
        self.accuracy_val = None
        self.accuracy_test = None
        self.accuracy_hist = None
        self.loss_hist = None

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)
        x = self.linear_1(x)
        x = self.activation_fct_lin_1(x)

        x = self.linear_2(x)
        x = self.activation_fct_lin_2(x)

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

    def output_0(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 0].reshape(len(x), 1).detach().numpy()

    def output_0_torch(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 0].reshape(len(x), 1)

    def output_1(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 1].reshape(len(x), 1).detach().numpy()

    def output_1_torch(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 1].reshape(len(x), 1)

    def output_2(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 2].reshape(len(x), 1).detach().numpy()

    def output_2_torch(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 2].reshape(len(x), 1)

    def output_3(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 3].reshape(len(x), 1).detach().numpy()

    def output_3_torch(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 3].reshape(len(x), 1)

    def output_4(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 4].reshape(len(x), 1).detach().numpy()

    def output_4_torch(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 4].reshape(len(x), 1)

    def output_5(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 5].reshape(len(x), 1).detach().numpy()

    def output_5_torch(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 5].reshape(len(x), 1)

    def output_6(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 6].reshape(len(x), 1).detach().numpy()

    def output_6_torch(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 6].reshape(len(x), 1)

    def output_7(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 7].reshape(len(x), 1).detach().numpy()

    def output_7_torch(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 7].reshape(len(x), 1)

    def output_8(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 8].reshape(len(x), 1).detach().numpy()

    def output_8_torch(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 8].reshape(len(x), 1)

    def output_9(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 9].reshape(len(x), 1).detach().numpy()

    def output_9_torch(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 9].reshape(len(x), 1)

    def output_10(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 10].reshape(len(x), 1).detach().numpy()

    def output_10_torch(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 10].reshape(len(x), 1)

    def output_11(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 11].reshape(len(x), 1).detach().numpy()

    def output_11_torch(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 11].reshape(len(x), 1)

    def output_12(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 12].reshape(len(x), 1).detach().numpy()

    def output_12_torch(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 12].reshape(len(x), 1)

    def output_13(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 13].reshape(len(x), 1).detach().numpy()

    def output_13_torch(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 13].reshape(len(x), 1)

    def output_14(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 14].reshape(len(x), 1).detach().numpy()

    def output_14_torch(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 14].reshape(len(x), 1)

    def output_15(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 15].reshape(len(x), 1).detach().numpy()

    def output_15_torch(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 15].reshape(len(x), 1)

    def output_16(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 16].reshape(len(x), 1).detach().numpy()

    def output_16_torch(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 16].reshape(len(x), 1)

    def output_17(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 17].reshape(len(x), 1).detach().numpy()

    def output_17_torch(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 17].reshape(len(x), 1)

    def output_18(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 18].reshape(len(x), 1).detach().numpy()

    def output_18_torch(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 18].reshape(len(x), 1)

    def output_19(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 19].reshape(len(x), 1).detach().numpy()

    def output_19_torch(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).clone().to(torch.float32)

        x = self.forward(x)

        return x[:, 19].reshape(len(x), 1)