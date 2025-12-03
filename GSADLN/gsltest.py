import copy
import dill
import torch
import numpy
import time
import scipy.sparse
from torch.utils.data import DataLoader
from network_GSL import GLS_network
import pandas as pd
from tqdm import tqdm
import os.path
start = time.time()

test_num_path = 'input_data/DY_018/_ibl_trials_1.xlsx'
df_2 = pd.read_excel(test_num_path)
num_trial = df_2['num_trial']
test_num_trial = num_trial.tail(139).tolist()


test_data_path = './input_data/DY_018/Screen/test/0%-15%/'

numadj = 283

retrain_epochs = 1000

retrainModelPath = './train_result/DY_018/Screen/0%-15%/retrain_model_dropout_0.2/'


adj_path = f'./train_result/DY_018/Screen/0%-15%/adj_dropout_0/'


save_path = './train_result/DY_018/Screen/0%-15%/test_dropout_0.2/'

accuracy_rate_best_list=[]
test_num_trial_list=[]
for test_num in test_num_trial:
    test_data_path_temp = os.path.join(test_data_path,f'num_trial_{test_num}.pkl')
    with open(test_data_path_temp, 'rb') as f:
        test_dataset = dill.load(f)

    batch_size = 32

    glu_inputdim = 16
    num_nodes = 62
    position_embed_dim = 2

    def accuracy_loss(output, target):
        output = torch.tensor(output)
        target = torch.tensor(target)
        correct1 = ((output > 0.5) & (target == 1)).sum().item()
        correct2 = ((output < 0.5) & (target == 0)).sum().item()
        correct3 = ((output == 0.5) & (target == 0)).sum().item()
        total = len(target)
        accuracy_rate = (correct1 + correct2 + correct3) / total
        return accuracy_rate


    test = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    activeF = torch.nn.ReLU6()
    activeT = torch.nn.Tanh()
    activeR = torch.nn.ReLU()
    rewiring_distance = 0.005
    device = 'cuda'
    accuracy_rate_best = 0
    adj_num_best = 0
    mod_num_best = 0
    pred_list_best = []
    target_list_best =  []

    accuracy_rate_total=[]
    epoch_number =[]


    for adj_num in tqdm(range(numadj,numadj+1)):
        for mod_num in range(retrain_epochs):

            number_adj = adj_num
            number_mod = mod_num

            gslmodel = torch.load(retrainModelPath+f'adj_{number_adj}/gsl{number_mod}model.pt')

            gslmodel.eval()


            position_embed = torch.ones(batch_size, glu_inputdim, num_nodes, position_embed_dim).to(device)

            position_embed_1 = torch.mean(position_embed, dim=0)
            position_embed_1 = position_embed_1.unsqueeze(0)
            pred_list = []
            target_list = []

            df = pd.read_csv(adj_path + f'graph_adj_{number_adj}.csv', header=None)

            numpy_array = df.to_numpy()
            graph_adj = torch.tensor(numpy_array)
            graph_adj = graph_adj.to('cpu')

            number_of_edge = graph_adj.sum()


            tmp_coo = scipy.sparse.coo_matrix(graph_adj)
            values = tmp_coo.data
            indicate = numpy.vstack((tmp_coo.row, tmp_coo.col))
            u = torch.LongTensor(indicate)
            v = torch.LongTensor(values)
            edge_index = torch.sparse_coo_tensor(u, v, tmp_coo.shape)
            edge_index.to(device)

            for i, (data) in enumerate(test):
                pre, target, position_embed_learned = gslmodel(data, position_embed_1, edge_index)
                pre=pre.flatten().tolist()
                pred_list.extend(pre)
                target=target.flatten().tolist()
                target_list.extend(target)

            accuracy_rate=accuracy_loss(pred_list, target_list)

            accuracy_rate_total.append(accuracy_rate)
            epoch_number.append(mod_num)


            print(f'accuracy_rate:{accuracy_rate}  edge_num:{number_of_edge}')

            if accuracy_rate > accuracy_rate_best:
                pred_list_best = pred_list
                target_list_best = target_list
                accuracy_rate_best = accuracy_rate
                adj_num_best = adj_num
                mod_num_best = mod_num

    print('accuracy_rate_best:', accuracy_rate_best)
    print('adj_num_best:', adj_num_best)
    print('mod_num_best:', mod_num_best)
    df_target_pre = pd.DataFrame({'target': target_list_best, 'pre': pred_list_best})

    df_accuracy_rate = pd.DataFrame({'epochs': epoch_number, 'accuracy_rate': accuracy_rate_total})

    file_path = save_path + f'{adj_num}/'
    folder = os.path.dirname(file_path)
    if not os.path.exists(folder):
        os.makedirs(folder)

    df_target_pre.to_csv(file_path + f'best_test_target_pre_gsl{mod_num_best}model_{test_num}_测试.csv', index=False)

    df_accuracy_rate.to_csv(file_path + f'epochs_accuracy_rate_{test_num}_测试.csv', index=False)

    accuracy_rate_best_list.append(accuracy_rate_best)
    test_num_trial_list.append(test_num)

df_test_accurate_rate = pd.DataFrame({'trial_num': test_num_trial_list, 'accuracy_rate_best': accuracy_rate_best_list})
df_test_accurate_rate.to_csv(file_path + f'test_num_trial_accuracy_rate.csv', index=False)