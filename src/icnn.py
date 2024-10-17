import torch
from torch import nn, Tensor
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

from src.parameters import *

class customLinear(torch.nn.Module):

    def __init__(self, size_in, size_out, weights):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        self.weights = weights.reshape([len(weights), size_out, size_in])

    def forward(self, x):
        w_times_x = torch.bmm(x, self.weights.transpose(-2, -1))
        return w_times_x

class customConvexLinear(torch.nn.Module):

    def __init__(self, size_in, size_out, weights):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        self.weights = weights.reshape([len(weights), size_out, size_in])

    def forward(self, x):
        w_times_x = torch.bmm(x, torch.nn.functional.softplus(self.weights.transpose(-2, -1)))
        return w_times_x

def symm_softplus(x, softplus_=torch.nn.functional.softplus):
    return softplus_(x) - 0.5 * x

def softplus(x):
    return nn.functional.softplus(x)

def squared_softplus(x):
    return torch.square(torch.nn.functional.softplus(x))

def activation_shifting(activation):
    def shifted_activation(x):
        return activation(x) - activation(torch.zeros_like(x))
    return shifted_activation

def get_softplus(softplus_type, zero_softplus=False):
    if softplus_type == 'softplus':
        act = nn.functional.softplus
    elif softplus_type == 'squared_softplus':
        act = squared_softplus
    else:
        raise NotImplementedError(
            f'softplus type {softplus_type} not supported.')
    if zero_softplus:
        act = activation_shifting(act)
    return act


class Softplus(nn.Module):
    def __init__(self, softplus_type, zero_softplus=False):
        super(Softplus, self).__init__()
        self.softplus_type = softplus_type
        self.zero_softplus = zero_softplus

    def forward(self, x):
        return get_softplus(self.softplus_type, self.zero_softplus)(x)


class SymmSoftplus(torch.nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, x):
        return symm_softplus(x)


class PosLinear(torch.nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        gain = 1 / x.size(1)
        return nn.functional.linear(x, torch.nn.functional.softplus(self.weight)) * gain

class PosLinear2(torch.nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.linear(x, torch.softmax(self.weight, 1), self.bias)

class multi_task_learner(nn.Module):
    def __init__(self, graph_t_dim, learner_dim, fully_learner_dim, skip_learner_dim, weights_output_dim, skip_weights_output_dim, activation_func):
        super(multi_task_learner, self).__init__()
        self.dropout_p = 0.
        self.weights_output_dim = weights_output_dim
        self.skip_weights_output_dim = skip_weights_output_dim
        self.out_dim = sum(self.weights_output_dim)
        self.skip_out_dim = sum(self.skip_weights_output_dim)
        
        if activation_func == 'leaky_relu':
            self.act = nn.LeakyReLU()
        elif activation_func == 'gelu':
            self.act = nn.GELU()
        elif activation_func == 'celu':
            self.act == nn.CELU()
            
        self.hidden_dim = learner_dim
        self.fully_hidden_dim = fully_learner_dim
        self.skip_hidden_dim = skip_learner_dim

        self.model = torch.nn.Sequential()
        self.model.add_module('e_fc1', nn.Linear(
            graph_t_dim, self.hidden_dim[0], bias=True))
        self.model.add_module('e_act1', self.act)

        for i in range(1, len(self.hidden_dim)):
            self.model.add_module(
                'e_fc' + str(i+1), nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i], bias=True))
            self.model.add_module('e_act' + str(i+1), self.act)
        
        self.fully_model = torch.nn.Sequential()
        self.fully_model.add_module('f_fc1', nn.Linear(self.hidden_dim[-1], self.fully_hidden_dim[0]))

        for i in range(1, len(self.fully_hidden_dim)):
            self.fully_model.add_module('f_fc'+str(i+1), nn.Linear(self.fully_hidden_dim[i-1], self.fully_hidden_dim[i]))
            self.fully_model.add_module('f_act'+str(i+1), self.act)

        self.skip_model = torch.nn.Sequential()
        self.skip_model.add_module('s_fc1', nn.Linear(self.hidden_dim[-1], self.skip_hidden_dim[0]))

        for i in range(1, len(self.skip_hidden_dim)):
            self.skip_model.add_module('s_fc'+str(i+1), nn.Linear(self.skip_hidden_dim[i-1], self.skip_hidden_dim[i]))
            self.skip_model.add_module('s_act'+str(i+1), self.act)

        self.output_layer = nn.Linear(self.fully_hidden_dim[-1], self.out_dim, bias=True)
        self.skip_output_layer = nn.Linear(self.skip_hidden_dim[-1], self.skip_out_dim, bias=True)

    def initialize_kaiming_weights(self):
        for module in self.children():
            if isinstance(module, nn.Linear):
                # Choose the desired weight initialization function
                torch.nn.init.kaiming_uniform_(
                    module.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):

        z = self.model(x)

        fully_z = self.fully_model(z)
        skip_z = self.skip_model(z)

        out = self.output_layer(fully_z)
        skip_out = self.skip_output_layer(skip_z)

        weights_out = []
        skip_weights_out = []

        dim1 = 0
        for count in range(len(self.weights_output_dim)):
            if count == 0:
                weights_out.append(out[:, :self.weights_output_dim[0]])
            else:
                weights_out.append(
                    out[:, dim1:dim1+self.weights_output_dim[count]])
            dim1 += self.weights_output_dim[count]

        dim1 = 0
        for count in range(len(self.skip_weights_output_dim)):
            if count == 0:
                skip_weights_out.append(skip_out[:, :self.skip_weights_output_dim[0]])
            else:
                skip_weights_out.append(
                    skip_out[:, dim1:dim1+self.skip_weights_output_dim[count]])
            dim1 += self.skip_weights_output_dim[count]

        return weights_out, skip_weights_out

class constitutiveNN(nn.Module):
    def __init__(self, c_hidden_dim, activation_func, layer_type, scaling_sftpSq):
        super(constitutiveNN, self).__init__()
        self.dropout_p = 0.

        if activation_func == 'squared_softplus':
            self.act = Softplus(softplus_type='squared_softplus', zero_softplus=False)

        self.scaling_sftpSq = scaling_sftpSq

        self.c_hidden_dim = c_hidden_dim
        self.layer_type = layer_type

        # dummy layers used to get the dimension for the weights of each layer and the skip connections
        # these layers are not used in the forward pass 

        self.dummy_layers = torch.nn.ModuleDict()
        self.dummy_skip_layers = torch.nn.ModuleDict()
        self.depth = len(self.c_hidden_dim)

        self.dummy_layers[str(0)] = torch.nn.Linear(ICNN_n_input, self.c_hidden_dim[0], bias = False).float()

        for i in range(1, self.depth):
            self.dummy_layers[str(i)] = torch.nn.Linear(self.c_hidden_dim[i-1], self.c_hidden_dim[i], bias = False).float()

            self.dummy_skip_layers[str(i)] = torch.nn.Linear(ICNN_n_input, self.c_hidden_dim[i], bias = False).float()

        self.dummy_layers[str(self.depth)] = torch.nn.Linear(self.c_hidden_dim[self.depth-1], ICNN_n_output, bias = False).float()

        self.dummy_skip_layers[str(self.depth)] = torch.nn.Linear(ICNN_n_input, ICNN_n_output, bias = False).float()

    def forward(self, x_input, weights_pred, skip_weights_pred):

        self.layers = torch.nn.ModuleDict()
        self.skip_layers = torch.nn.ModuleDict()
        self.depth = len(self.c_hidden_dim)

        weights_count = 0
        skip_weights_count = 0
        self.layers[str(0)] = customLinear(ICNN_n_input, self.c_hidden_dim[0], weights_pred[0]).float()
        weights_count += 1

        for i in range(1, self.depth):
            self.layers[str(i)] = customConvexLinear(self.c_hidden_dim[i-1], self.c_hidden_dim[i], weights_pred[weights_count]).float()
            weights_count += 1

            self.skip_layers[str(i)] = customLinear(ICNN_n_input, self.c_hidden_dim[i], skip_weights_pred[skip_weights_count]).float()
            skip_weights_count += 1

        self.layers[str(self.depth)] = customConvexLinear(self.c_hidden_dim[self.depth-1], ICNN_n_output, weights_pred[weights_count]).float()
        weights_count += 1

        self.skip_layers[str(self.depth)] = customLinear(ICNN_n_input, ICNN_n_output, skip_weights_pred[skip_weights_count]).float()

        x_input = x_input.unsqueeze(1)
        z = x_input.clone()
        z = self.layers[str(0)](z)

        for layer in range(1,self.depth):
            skip = self.skip_layers[str(layer)](x_input)
            z = self.layers[str(layer)](z)
            z += skip
            z = torch.nn.functional.softplus(z)
            z = self.scaling_sftpSq*torch.square(z)
        y = self.layers[str(self.depth)](z) + self.skip_layers[str(self.depth)](x_input)

        return y.squeeze(1)
