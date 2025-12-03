import copy
import dill
import torch

print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.__version__)


import numpy
import time
import scipy.sparse
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.neighbors import kneighbors_graph
from network_GSL import GLS_network
import os.path

print(torch.cuda.is_available())


numpy.random.seed(2024)
torch.manual_seed(2024)


start = time.time()

with open('./input_data/DY_018/Screen/train/0%-15%_前11中的后10个训练.pkl', 'rb') as f:
    train_dataset = dill.load(f)


file_path = f'./train_result/DY_018/Screen/0%-15%/train_model_dropout_0/'
folder = os.path.dirname(file_path)
if not os.path.exists(folder):
    os.makedirs(folder)


epochs = 500
lr = 0.0001




batch_size = 32

glu_inputdim = 16
num_nodes = 62
position_embed_dim = 2
rewiring_distance = 0.005
knn_num = 5
input_dim = 1
seq_length = 30
optim_graph_test_count = 2
gat_head = 8
gat_out_dim = 3
glu_output_dim = 32
dcn_layer_num = 2
device = 'cuda'

glsmodel = GLS_network(nodes_num=num_nodes, position_embed_dim=position_embed_dim, input_dim=input_dim,
                       seq_length=seq_length,
                       glu_inputdim=glu_inputdim, dcn_layers=dcn_layer_num, output_dim_glu=glu_output_dim,
                       gat_head=gat_head,
                       gat_outdim=glu_output_dim, dropout_rate=0, device_type=device)

optimizer = torch.optim.Adam(glsmodel.parameters(), lr=lr, weight_decay=0)
loss_function = torch.nn.BCELoss().cuda()
train = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


train_loss = []
test_loss_list = []

activeF = torch.nn.ReLU6()
activeT = torch.nn.Tanh()
activeR = torch.nn.ReLU()

position_embed = torch.ones(batch_size, glu_inputdim, num_nodes, position_embed_dim).to(device)
best_loss_count = 0
position_vector_learned = 0


for epoch in range(epochs):
    glsmodel.train()
    loss_epoch = 0
    epoch_start = time.time()
    mask = torch.eye(num_nodes, num_nodes).bool().to(device)
    Bgraph = torch.mean(position_embed, dim=1)
    Bgraph = torch.mean(Bgraph, dim=0)
    graph_adj = torch.norm(Bgraph[:, None] - Bgraph, dim=2, p=2)
    graph_adj = activeF(graph_adj)
    graph_adj_1 = graph_adj
    graph_adj = torch.where(graph_adj < rewiring_distance, 1.0, 0.0)
    #print(graph_adj.shape)
    graph_adj.masked_fill_(mask, 0)
    number_of_edge = graph_adj.sum()
    print(f'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Epoch:{epoch}    number_of_edge :  {number_of_edge}')



    graph_adj = graph_adj.to('cpu')
    tmp_coo = scipy.sparse.coo_matrix(graph_adj)
    values = tmp_coo.data
    indicate = numpy.vstack((tmp_coo.row, tmp_coo.col))
    u = torch.LongTensor(indicate)
    v = torch.LongTensor(values)
    edge_index = torch.sparse_coo_tensor(u, v, tmp_coo.shape)
    edge_index.to(device)
    graph_adj = graph_adj.to(device)


    for i, (data) in enumerate(train):

        optimizer.zero_grad()
        pre, target, position_embed_learned = glsmodel(data, position_embed, edge_index)

        loss = loss_function(pre, target)
        loss_epoch = loss_epoch + copy.deepcopy(float(loss))
        loss.backward(retain_graph=False)
        optimizer.step()


    if epoch == 0:
        best_loss = loss_epoch
        print(position_embed_learned)
        position_embed = position_embed_learned.detach()
        current_best_epoch = epoch
        print('Graph Structure optimizing !')
    else:

        if loss_epoch < best_loss:
            best_loss = loss_epoch
            position_embed = position_embed_learned.detach()
            current_best_epoch = epoch

        else:
            best_loss = best_loss
            position_embed = position_embed.detach()
            best_loss_count = best_loss_count + 1

    print('Current epoch' + str(epoch) + 'loss = ' + str(best_loss))




    if best_loss_count < optim_graph_test_count:

        best_position_embed = position_embed
        print('Graph Structure Optimizing !')

    elif best_loss_count == optim_graph_test_count:
        print('Graph Structure optimizing complete !')
        best_position_embed = position_embed
        position_embed = best_position_embed.detach()

        torch.save(best_position_embed,
                   file_path + 'best_position_embed.pt')

        print('Optimal graph structure train loss is ' + str(best_loss))
        print('#########################Current best epoch:',current_best_epoch+1 )
        best_loss_count = 0
    else:
        position_embed = position_embed.detach()


    epoch_end = time.time()
    print('epoch time consume ' + str(epoch_end - epoch_start) + ' s')

    torch.save(graph_adj_1, file_path
               + 'glsinitial' + str(epoch) + 'adj.pt')

print("end")


