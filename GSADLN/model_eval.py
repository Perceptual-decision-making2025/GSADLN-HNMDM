import copy
import dill
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from network_model import ConvLSTM, Net, ConvLSTMCell, G_GAT
import csv


accuracy_each = []

with open('test_dataset_save.pkl', 'rb') as f:
    test_dataset = dill.load(f)

test = DataLoader(dataset=test_dataset, batch_size=113, shuffle=False, collate_fn=lambda x: x)

model = torch.load('9model_trained.pt')
model.eval()
pred_list = []
target_list = []

pre_tensor = torch.Tensor()
target_tensor = torch.Tensor()

for i, (data) in enumerate(test):

    for j in range(len(data)):
        data_id = data[j]

        data_id = data_id.cuda()
        out, rul = model(data_id)

        out = torch.as_tensor(out).to(torch.float32)
        out = torch.sigmoid(out*1000)

        rul = torch.as_tensor(rul).to(torch.float32)

        out_values = out.tolist()
        rul_values = rul.tolist()
        rul_values1 = rul_values[0]
        rul_values = rul_values1[:30]
        out_values =out_values[0]

        threshold = 0.5
        pred_list_t = torch.tensor(out_values)
        target_list_t = torch.tensor(rul_values)
        binary_prediction = (pred_list_t >= threshold).float()

        correct_prediction = (binary_prediction == target_list_t).sum().item()
        total_prediction = binary_prediction.numel()
        accuracy = correct_prediction / total_prediction
        accuracy_each.append((copy.deepcopy(accuracy)))



        for i in range(30):
            pred_list.append(copy.deepcopy(out_values[i]))
            target_list.append(copy.deepcopy(rul_values[i]))
threshold = 0.5
pred_list_t = torch.tensor(pred_list)
target_list_t = torch.tensor(target_list)
binary_prediction = (pred_list_t >= threshold).float()

correct_prediction = (binary_prediction == target_list_t).sum().item()
total_prediction = binary_prediction.numel()
accuracy = correct_prediction / total_prediction
print(f'accuracy:{accuracy: .2%}')


print(accuracy_each)


plt.figure()

plt.plot(accuracy_each, 'r--', label='accuracy_each')
plt.legend()
plt.show()
