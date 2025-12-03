import torch
import torch_geometric
import copy
import matplotlib.pyplot as plt
import dill
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import time



class G_GAT(torch.nn.Module):
    def __init__(self, input_feature_channel, output_feature_channel):
        super(G_GAT, self).__init__()
        self.conv1 = torch_geometric.nn.GATConv(in_channels=input_feature_channel, out_channels=output_feature_channel,
                                                heads=33, concat=False, dropout=0.2)


    def forward(self, sensor_x, sensor_edge_index):
        x =  sensor_x
        edge_index = sensor_edge_index
        x = self.conv1(x, edge_index)



        x = torch.tanh(x)

        return x




class ConvLSTMCell(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = torch.nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                    out_channels=4 * self.hidden_dim,
                                    kernel_size=self.kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state


        input_tensor = input_tensor.repeat(1, 1, 1, 1)

        combined = torch.cat([input_tensor, h_cur], dim=1)


        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.nn.init.xavier_normal_(
            torch.rand(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)),
                torch.nn.init.xavier_normal_(
                    torch.rand(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)))


class ConvLSTM(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)


        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = torch.nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):

        if not self.batch_first:

            input_tensor = input_tensor.permute(1,0,2)


        b,_, h, w = input_tensor.size()


        if hidden_state is not None:
            raise NotImplementedError()
        else:

            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t,  :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param



class Net(torch.nn.Module):
    def __init__(self, step_length, gat_output_seq_length, sensor_edge_index, channel_num,
                 hidden_size, lstm_output_length):
        super(Net, self).__init__()
        self.gat_input_size = step_length
        self.gat_output_size = gat_output_seq_length
        self.sensor_edge_index = sensor_edge_index

        self.lstm_input_size = channel_num
        self.lstm_hidden_size = hidden_size
        self.lstm_output_size = lstm_output_length
        self.layer1g = G_GAT(self.gat_input_size, self.gat_output_size)


        self.layer3 = ConvLSTM(input_dim=1, hidden_dim=self.lstm_hidden_size, kernel_size=(5, 5),
                               num_layers=1, batch_first=True, bias=True, return_all_layers=False)



        self.fc1 = torch.nn.Linear(self.lstm_hidden_size*40*30, 30)

        self.fc2 = torch.nn.Linear(64, self.lstm_output_size)#64
        self.dropout1d = torch.nn.Dropout(p=0.2)
        self.dropout2d = torch.nn.Dropout(p=0.2)

    def forward(self, x):
        gat_out_tensor = torch.Tensor()
        gat_out_tensor = gat_out_tensor.cuda()
        rul_list = list()
        for k in range(x.size(1)):
            cycle_data = x[0:5, k:k + 1]
            g_graph = x[0:40, k:k + 1]
            g_graph = g_graph.to(torch.float32)
            g_out = self.layer1g(g_graph, self.sensor_edge_index)
            g_out = g_out.permute(1, 0)


            gat_out_tensor = torch.cat((gat_out_tensor, g_out), dim=-1)
        rul_target = x[40:41, :]
        gat_out_tensor = torch.unsqueeze(gat_out_tensor, -1)
        gat_out_tensor = gat_out_tensor.permute(2, 0, 1)

        gat_out_tensor = self.dropout2d(gat_out_tensor)

        gat_out_tensor = torch.unsqueeze(gat_out_tensor, -1)
        gat_out_tensor = gat_out_tensor.permute(0, 1, 2, 3)
        self.layer3.to('cuda')
        gat_out_tensor_GPU = gat_out_tensor.cuda()

        output = self.layer3(gat_out_tensor_GPU)

        self.layer3.to('cuda')
        gat_out_tensor_GPU = gat_out_tensor.cuda()

        output = self.layer3(gat_out_tensor_GPU)


        prediction = output[0][0]

        prediction = prediction[:, -1:, :, :]
        prediction = prediction.reshape(1, -1)
        prediction = torch.tanh(prediction)
        prediction = self.dropout1d(prediction)

        prediction = self.fc1(prediction)


        return prediction, rul_target
