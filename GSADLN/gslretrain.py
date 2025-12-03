import copy
import os.path

import dill
import torch
import numpy
import time
import scipy.sparse
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.neighbors import kneighbors_graph
from network_GSL import GLS_network
from tqdm import tqdm
from torch.optim import lr_scheduler



numpy.random.seed(2024)
torch.manual_seed(2024)

start = time.time()

with open('./input_data/DY_018/Screen/train/0%-15%_前11中的后10个训练.pkl', 'rb') as f: # 打开文件赋予f
    train_dataset = dill.load(f)




numadj = 283

adj_path = f'train_result/DY_018/Screen/0%-15%/adj_dropout_0/'

save_path = 'train_result/DY_018/Screen/0%-15%/retrain_model_dropout_0.2/'

epochs = 1000
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


best_loss_count = 0
position_vector_learned = 0


for adj_num in tqdm(range(numadj,numadj+1)):
    glsmodel = GLS_network(nodes_num=num_nodes, position_embed_dim=position_embed_dim, input_dim=input_dim,
                           seq_length=seq_length,
                           glu_inputdim=glu_inputdim, dcn_layers=dcn_layer_num, output_dim_glu=glu_output_dim,
                           gat_head=gat_head,
                           gat_outdim=glu_output_dim, dropout_rate=0.2, device_type=device)

    optimizer = torch.optim.Adam(glsmodel.parameters(), lr=lr, weight_decay=0)
    loss_function = torch.nn.BCELoss().cuda() # torch.nn.MSELoss().cuda()
    train = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    train_loss = []
    test_loss_list = []

    activeF = torch.nn.ReLU6()
    activeT = torch.nn.Tanh()
    activeR = torch.nn.ReLU()

    position_embed = torch.ones(batch_size, glu_inputdim, num_nodes, position_embed_dim).to(device)


    number_adj = adj_num

    df = pd.read_csv(adj_path +f'graph_adj_{number_adj}.csv')
    numpy_array = df.to_numpy()
    graph_adj = torch.tensor(numpy_array)
    graph_adj = graph_adj.to('cpu')
    tmp_coo = scipy.sparse.coo_matrix(graph_adj)
    values = tmp_coo.data
    indicate = numpy.vstack((tmp_coo.row, tmp_coo.col))
    u = torch.LongTensor(indicate)
    v = torch.LongTensor(values)
    edge_index = torch.sparse_coo_tensor(u, v, tmp_coo.shape)
    edge_index.to(device)
    graph_adj = graph_adj.to(device)

    for epoch in range(epochs):
        # adjust_learning_rate(optimizer, epoch, lr)
        glsmodel.train()
        loss_epoch = 0
        epoch_start = time.time()

        for i, (data) in enumerate(train):
            # enumerate函数会按照循环分别取出每一个batch的数据，返回一个数据列表
            optimizer.zero_grad()
            pre, target, position_embed_learned = glsmodel(data, position_embed, edge_index)

            loss = loss_function(pre, target)
            loss_epoch = loss_epoch + copy.deepcopy(float(loss))
            loss.backward(retain_graph=False)
            optimizer.step()

        print('Current epoch' + str(epoch) + 'loss = ' + str(loss_epoch))

print("end")


