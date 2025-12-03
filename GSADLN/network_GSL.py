import torch
import numpy
import torch_geometric
import math
from torch.nn.utils import weight_norm
import torch.nn.functional as F




class Casual_GLU(torch.nn.Module):


    def __init__(self, in_channels, out_channels, timeblock_padding, layers_dcn, dropout, kernel_size,
                 start_dilation=1):
        super(Casual_GLU, self).__init__()
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.timeblock_padding = timeblock_padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DCN_l1 = DCN(in_dim=in_channels, out_dim=out_channels,
                          kernel_size=kernel_size, layers=layers_dcn, timeblock_padding=timeblock_padding,
                          dilation=start_dilation)
        self.DCN_l2 = DCN(in_dim=in_channels, out_dim=out_channels,
                          kernel_size=kernel_size, layers=layers_dcn, timeblock_padding=timeblock_padding,
                          dilation=start_dilation)
        self.DCN_l3 = DCN(in_dim=in_channels, out_dim=out_channels,
                          kernel_size=kernel_size, layers=layers_dcn, timeblock_padding=timeblock_padding,
                          dilation=start_dilation)

    def forward(self, x):
        res = self.DCN_l3(x)
        out = torch.tanh(self.DCN_l1(x)) * torch.sigmoid(self.DCN_l2(x))

        out = F.dropout(out, self.dropout, training=self.training)
        out = F.relu(out + res)

        return out


class DCN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, layers, timeblock_padding, dilation):
        super(DCN, self).__init__()
        self.layers = layers
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.timeblock_padding = timeblock_padding
        self.filter_convs = torch.nn.ModuleList()

        new_dilation = dilation
        for i in range(layers):

            if i == 0:
                self.filter_convs.append(weight_norm(torch.nn.Conv2d(in_channels=in_dim, out_channels=out_dim,
                                                                     kernel_size=(1, kernel_size),
                                                                     dilation=new_dilation)))
            else:
                self.filter_convs.append(weight_norm(torch.nn.Conv2d(in_channels=out_dim, out_channels=out_dim,
                                                                     kernel_size=(1, kernel_size),
                                                                     dilation=new_dilation)))
            new_dilation = new_dilation * 2
        self.receptive_field = numpy.power(2, layers) - 1

    def forward(self, x):

        if self.in_dim == self.out_dim:
            out = x
        else:
            pad_num = self.kernel_size - 1
            out = x
            for i in range(self.layers):
                out = F.pad(out, (pad_num, 0, 0, 0))
                out = self.filter_convs[i](out)
                pad_num = pad_num * 2

        if self.timeblock_padding:
            out = out
        else:
            out = out[:, :, :, self.receptive_field:out.shape[3]]
        return out



class GLS_network(torch.nn.Module):
    def __init__(self, nodes_num, position_embed_dim, input_dim, seq_length, glu_inputdim, output_dim_glu, dcn_layers,
                 gat_head,
                 gat_outdim, dropout_rate, device_type):
        super(GLS_network, self).__init__()
        self.nodes_num = nodes_num
        self.position_dim = position_embed_dim
        self.input_dim = input_dim
        self.device = device_type
        self.output_dim_glu = output_dim_glu
        self.dcn_layers = dcn_layers
        self.dropout_glu = dropout_rate
        self.input_dim_GLU = glu_inputdim
        self.gat_out_dim = gat_outdim
        self.gat_head = gat_head
        self.seq_len = seq_length

        self.start_conv = weight_norm(torch.nn.Conv2d(in_channels=self.input_dim, out_channels=self.input_dim_GLU,
                                                      kernel_size=(1, 1))).to(self.device)

        self.GLU = Casual_GLU(in_channels=self.input_dim_GLU, out_channels=self.output_dim_glu,
                              layers_dcn=self.dcn_layers,
                              dropout=self.dropout_glu, kernel_size=2, start_dilation=1, timeblock_padding=True).to(self.device)

        self.gat_conv = torch_geometric.nn.GATv2Conv(in_channels=self.output_dim_glu * self.seq_len,
                                                     out_channels=self.gat_out_dim * self.seq_len,
                                                     heads=self.gat_head, concat=True, dropout=self.dropout_glu).to(self.device)
        self.fc1 = torch.nn.Linear(self.nodes_num, 1).to(self.device)
        self.fc2 = torch.nn.Linear(self.gat_out_dim * self.gat_head, 1).to(self.device)
        self.fc3 = torch.nn.Linear(self.output_dim_glu, self.input_dim_GLU).to(self.device)

    def forward(self, data, position_embed, edge_index):
        train_data = data[:, 0:62, :]
        target = data[:, 62:63, :].to(self.device)


        target = target.float()
        train_data = train_data.to(self.device).float()
        train_data = torch.unsqueeze(train_data, dim=1)

        edge_index = edge_index.to(self.device)

        x = self.start_conv(train_data)
        x = torch.cat((x, position_embed), dim=3)
        x = self.GLU(x)

        node_feature = x[:, :, :, :x.shape[3] - self.position_dim]
        position_embed_learned = x[:, :, :, x.shape[3] - self.position_dim:x.shape[3]]
        position_embed_learned = position_embed_learned.permute(0, 2, 3, 1)
        position_embed_learned = self.fc3(position_embed_learned)
        position_embed_learned = position_embed_learned.permute(0, 3, 1, 2)

        out = torch.Tensor().to(self.device)

        b=range(node_feature.shape[0])
        for i in range(node_feature.shape[0]):

            feature = node_feature[i, :, :, :]
            feature = feature.permute(1, 0, 2)
            feature = torch.reshape(feature, (self.nodes_num, self.output_dim_glu * self.seq_len))
            sub_out = self.gat_conv(feature, edge_index._indices())

            sub_out = torch.unsqueeze(sub_out, 0)
            sub_out = sub_out.reshape(1, self.nodes_num, self.gat_out_dim * self.gat_head, self.seq_len)
            out = torch.cat((out, sub_out), dim=0)

        pre = out.permute(0, 3, 1, 2)
        pre = self.fc2(pre)
        pre = torch.squeeze(pre, -1)
        pre = self.fc1(pre)
        pre = torch.squeeze(pre, -1)
        pre = pre.float()
        pre = torch.unsqueeze(pre, 1)
        pre = F.sigmoid(pre)

        return pre, target, position_embed_learned


class reGLS_network(torch.nn.Module):
    def __init__(self, nodes_num, position_embed_dim, input_dim, seq_length, glu_inputdim, output_dim_glu, dcn_layers,
                 gat_head,
                 gat_outdim, dropout_rate, device_type):
        super(reGLS_network, self).__init__()
        self.nodes_num = nodes_num
        self.position_dim = position_embed_dim
        self.input_dim = input_dim
        self.device = device_type
        self.output_dim_glu = output_dim_glu
        self.dcn_layers = dcn_layers
        self.dropout_glu = dropout_rate
        self.input_dim_GLU = glu_inputdim
        self.gat_out_dim = gat_outdim
        self.gat_head = gat_head
        self.seq_len = seq_length

        self.start_conv = weight_norm(torch.nn.Conv2d(in_channels=self.input_dim, out_channels=self.input_dim_GLU,
                                                      kernel_size=(1, 1))).to(self.device)

        self.GLU = Casual_GLU(in_channels=self.input_dim_GLU, out_channels=self.output_dim_glu,
                              layers_dcn=self.dcn_layers,
                              dropout=self.dropout_glu, kernel_size=2, start_dilation=1, timeblock_padding=True).to(self.device)

        self.gat_conv = torch_geometric.nn.GATv2Conv(in_channels=self.output_dim_glu * self.seq_len,
                                                     out_channels=self.gat_out_dim * self.seq_len,
                                                     heads=self.gat_head, concat=True, dropout=self.dropout_glu).to(self.device)
        self.fc1 = torch.nn.Linear(self.nodes_num, 1).to(self.device)
        self.fc2 = torch.nn.Linear(self.gat_out_dim * self.gat_head, 1).to(self.device)
        self.fc3 = torch.nn.Linear(self.output_dim_glu, self.input_dim_GLU).to(self.device)

    def forward(self, data, position_embed, edge_index):
        train_data = data[:, 0:62, :]
        target = data[:, 62:63, :].to(self.device)


        target = target.float()
        train_data = train_data.to(self.device).float()
        train_data = torch.unsqueeze(train_data, dim=1)
        edge_index = edge_index.to(self.device)

        x = self.start_conv(train_data)
        x = torch.cat((x, position_embed), dim=3)
        x = self.GLU(x)

        node_feature = x[:, :, :, :x.shape[3] - self.position_dim]
        position_embed_learned = x[:, :, :, x.shape[3] - self.position_dim:x.shape[3]]
        position_embed_learned = position_embed_learned.permute(0, 2, 3, 1)
        position_embed_learned = self.fc3(position_embed_learned)
        position_embed_learned = position_embed_learned.permute(0, 3, 1, 2)

        out = torch.Tensor().to(self.device)
        b=range(node_feature.shape[0])
        for i in range(node_feature.shape[0]):

            feature = node_feature[i, :, :, :]
            feature = feature.permute(1, 0, 2)
            feature = torch.reshape(feature, (self.nodes_num, self.output_dim_glu * self.seq_len))
            sub_out = self.gat_conv(feature, edge_index._indices())

            sub_out = torch.unsqueeze(sub_out, 0)
            sub_out = sub_out.reshape(1, self.nodes_num, self.gat_out_dim * self.gat_head, self.seq_len)
            out = torch.cat((out, sub_out), dim=0)

        pre = out.permute(0, 3, 1, 2)
        pre = self.fc2(pre)
        pre = torch.squeeze(pre, -1)
        pre = self.fc1(pre)
        pre = torch.squeeze(pre, -1)
        pre = pre.float()
        pre = torch.unsqueeze(pre, 1)
        pre = F.sigmoid(pre)

        return pre, target,