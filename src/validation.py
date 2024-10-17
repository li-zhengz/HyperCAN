from __future__ import print_function

import time
import numpy as np
import torch
import torch.nn as nn
import logging
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
import pickle

from utils import *
from src.parameters import *
from icnn import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_dir = '../model_checkpoint'

S_name = ['S11', 'S12', 'S13', 'S21', 'S22', 'S23', 'S31', 'S32', 'S33']
F_name = ['F11', 'F12', 'F13', 'F21', 'F22', 'F23', 'F31', 'F32', 'F33']

batch_size = 128

dropout = 0.
criterion = nn.MSELoss(reduction='sum')

#-----------------------------------------------------
######  Load model  ######
#-----------------------------------------------------
 
graph_model = torch.load(model_dir + '/graph_checkpoint.pth', map_location=torch.device('cpu')).to(device)
graph_model.eval()

icnn_model = torch.load(model_dir + '/icnn_checkpoint.pth', map_location=torch.device('cpu')).to(device)
icnn_model.eval()

#-----------------------------------------------------
######  Load data  ######
#-----------------------------------------------------
dataset_list = ['test_load_truss', 'test_load', 'test_truss', 'train']

for data in dataset_list:

    results_save_folder = f'../output/validation_results/{data}'
    os.makedirs(results_save_folder, exist_ok=True)

    training_dataset = torch.load(f'../data/dataset/{num_graphs}_{data}_dataset.pt')
    test_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        
    print("Validating Dataset", data, "...")

    y_label_list, y_pred_list, total_loss = [], [], 0
    F_list, W_label_list = [], []
    W_pred_list = []
    I_input_list = []
    Up_list = []
    E_list = []
    F0 = torch.eye(3).float().to(device)
    F0 = F0.reshape([1, 9])
    adj_list = []
    nodes_list = []

    for adj, nodes, F, W, S, mask, loadcase_idx in test_dataloader:

        F_input_raw = tensor_to_variable(F).view(-1, F.shape[-1])  # (batch_size, num_loadcase, 9)
        S_matrix_raw = tensor_to_variable(S).view(-1, S.shape[-1])
        W_label_raw = tensor_to_variable(W).view(-1, W.shape[-1])

        loadcase_idx_raw = tensor_to_variable(loadcase_idx).view(-1, loadcase_idx.shape[-1])

        adj_raw = tensor_to_variable(adj)
        nodes_raw = tensor_to_variable(nodes)

        graph_t = torch.cat((adj_raw, nodes_raw), dim=1)
        graph_t = tensor_to_variable(graph_t).unsqueeze(1).repeat(1, F.shape[1], 1).view(-1, graph_t.shape[-1])
        
        mask = tensor_to_variable(mask).view(-1, mask.shape[-1]).bool().squeeze()
        graph_t = graph_t[mask, :]
        F_input = F_input_raw[mask, :]
        S_matrix = S_matrix_raw[mask, :]
        W_label = W_label_raw[mask, :]
        loadcase_idx = loadcase_idx_raw[mask, :]

        F11 = F_input[:, 0]; F12 = F_input[:, 1]; F13 = F_input[:, 2]
        F21 = F_input[:, 3]; F22 = F_input[:, 4]; F23 = F_input[:, 5]
        F31 = F_input[:, 6]; F32 = F_input[:, 7]; F33 = F_input[:, 8]

        weights_out, skip_weights_out = graph_model(graph_t.float())
        
        C11, C12, C13, C21, C22, C23, C31, C32, C33 = computeCauchyGreenStrain(F11, F12, F13, F21, F22, F23, F31, F32, F33)
        
        E11, E12, E13, E21, E22, E23, E31, E32, E33 = computeGreenLagrangeStrain(C11, C12, C13, C21, C22, C23, C31, C32, C33)

        E11.requires_grad = True
        E12.requires_grad = True
        E13.requires_grad = True
        E21.requires_grad = True
        E22.requires_grad = True
        E23.requires_grad = True
        E31.requires_grad = True
        E32.requires_grad = True
        E33.requires_grad = True

        E_input = torch.stack((E11, E12, E13, E21, E22, E23, E31, E32, E33), dim=-1).float()

        F0_input = F0.repeat(len(W_label), 1).float().to(device)

        F0_11 = F0_input[:, 0]; F0_12 = F0_input[:, 1]; F0_13 = F0_input[:, 2]
        F0_21 = F0_input[:, 3]; F0_22 = F0_input[:, 4]; F0_23 = F0_input[:, 5]
        F0_31 = F0_input[:, 6]; F0_32 = F0_input[:, 7]; F0_33 = F0_input[:, 8]

        C0_11, C0_12, C0_13, C0_21, C0_22, C0_23, C0_31, C0_32, C0_33 = computeCauchyGreenStrain(
            F0_11, F0_12, F0_13, F0_21, F0_22, F0_23, F0_31, F0_32, F0_33)

        E0_11, E0_12, E0_13, E0_21, E0_22, E0_23, E0_31, E0_32, E0_33 = computeGreenLagrangeStrain(C0_11, C0_12, C0_13, C0_21, C0_22, C0_23, C0_31, C0_32, C0_33)

        E0_11.requires_grad = True
        E0_12.requires_grad = True
        E0_13.requires_grad = True
        E0_21.requires_grad = True
        E0_22.requires_grad = True
        E0_23.requires_grad = True
        E0_31.requires_grad = True
        E0_32.requires_grad = True
        E0_33.requires_grad = True

        E0_input = torch.stack((E0_11, E0_12, E0_13, E0_21, E0_22, E0_23, E0_31, E0_32, E0_33), dim=-1).float()

        W_NN = icnn_model(E_input.float(), weights_out, skip_weights_out)
        W0_NN = icnn_model(E0_input.float(), weights_out, skip_weights_out) 
        
        W_pred = W_NN - W0_NN

        dW_dE = torch.autograd.grad(
            W_NN, E_input, grad_outputs=torch.ones_like(W_NN), create_graph=True)[0]
        
        dW0_dE0 = torch.autograd.grad(
            W0_NN, E0_input, grad_outputs=torch.ones_like(W0_NN), create_graph=True)[0]

        S_pred = dW_dE - dW0_dE0
        S_label = S_matrix.reshape(F_input.shape)

        y_label_list.extend(variable_to_numpy(S_label))
        y_pred_list.extend(variable_to_numpy(S_pred))
        adj_list.extend(variable_to_numpy(adj_raw.unsqueeze(1).repeat(1, F.shape[1], 1).view(-1, adj.shape[-1])[mask,:]))
        nodes_list.extend(variable_to_numpy(nodes_raw.unsqueeze(1).repeat(1, F.shape[1], 1).view(-1, nodes.shape[-1])[mask,:]))
        F_list.extend(variable_to_numpy(F_input))
        W_label_list.extend(variable_to_numpy(W_label))
        W_pred_list.extend(variable_to_numpy(W_pred))

    adj_list = np.array(adj_list)
    nodes_list = np.array(nodes_list)
    y_label_list = np.array(y_label_list)
    y_pred_list = np.array(y_pred_list)
    F_list = np.array(F_list)
    W_label_list = np.array(W_label_list)
    W_pred_list = np.array(W_pred_list)

    print('*' * 20 + ' R2 score ' + '*' * 20)
    # Other components are small so we only look at R² values for the main components S11, S22, and S33
    S_norm = np.linalg.norm(y_label_list, axis=0)
    for col in [0,4,8]:  
        print(S_name[col], ' : ', f'{(r2_score(y_label_list[:,col], y_pred_list[:,col]))*100:.3f}%')
    print('*' * 20 + '   NRMSE  ' + '*' * 20)
    for col in range(len(S_name)):
        print(S_name[col], ' : ', f'{(calculate_nrmse(y_label_list[:,col], y_pred_list[:,col])):.3f}')

    filename = f'{results_save_folder}/S_label.pkl'
    pickle.dump(y_label_list, open(filename, "wb"))

    filename = f'{results_save_folder}/S_pred.pkl'
    pickle.dump(y_pred_list, open(filename, "wb"))

    filename = f'{results_save_folder}/F_label.pkl'
    pickle.dump(F_list, open(filename, "wb"))

    filename = f'{results_save_folder}/W_label.pkl'
    pickle.dump(W_label_list, open(filename, "wb"))

    filename = f'{results_save_folder}/W_pred.pkl'
    pickle.dump(W_pred_list, open(filename, "wb"))

    filename = f'{results_save_folder}/adj.pkl'
    pickle.dump(adj_list, open(filename, "wb"))

    filename = f'{results_save_folder}/nodes.pkl'
    pickle.dump(nodes_list, open(filename, "wb"))
