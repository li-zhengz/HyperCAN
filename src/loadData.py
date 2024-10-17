import os
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset
import random
from src.parameters import *
from utils import *

torch.multiprocessing.set_sharing_strategy('file_system')

dataset_type = 'train'

# dataset for evaluation
# dataset_type = 'test_truss': unseen truss structures 
# dataset_type = 'test_load': unseen load cases
# dataset_type = 'test_truss_load': unseen truss structures and unseen load cases

for dataset_type in ['train', 'test_truss', 'test_load', 'test_load_truss']:
    raw_data_folder = f'../data/FEM_data/{dataset_type}'
    save_folder = '../data/dataset'
    os.makedirs(save_folder, exist_ok=True)
        
    if dataset_type == 'train':
        num_loadcase = 14
    elif dataset_type == 'test_truss':
        num_loadcase = 13
    elif dataset_type == 'test_load':
        num_loadcase = 13
    elif dataset_type == 'test_truss_load':
        num_loadcase = 7

    adj_pkl_file = os.path.join(raw_data_folder, 'adj_data.pkl')
    node_pkl_file = os.path.join(raw_data_folder, 'nodes_data.pkl')

    F_pkl_file = os.path.join(raw_data_folder, 'F_data.pkl')
    W_pkl_file = os.path.join(raw_data_folder, 'W_data.pkl')
    S_pkl_file = os.path.join(raw_data_folder, 'S_data.pkl')

    Fs = pickle.load(open(F_pkl_file, "rb"))
    Fs = Fs.reshape([len(Fs), num_loadcase*num_load_step, 9])
    W_raw = pickle.load(open(W_pkl_file, "rb")).reshape([len(Fs), num_loadcase*num_load_step, 1])
    S_raw = pickle.load(open(S_pkl_file, "rb")).reshape([len(Fs) * num_loadcase*num_load_step, 9])

    # Reshape the array 'Fs' into a 2D array with rows corresponding to each combination of load case and load step,
    # and 9 columns representing stress components 
    Fs_vec = Fs.reshape([len(Fs) * num_loadcase * num_load_step, 9])
    ss_idx = np.where((Fs_vec[:,1] + Fs_vec[:,2] + Fs_vec[:,5] != 0))[0]                  # identify simple shear load cases
    tri_idx = np.where((Fs_vec[:,0] != 1) & (Fs_vec[:,4] != 1) & (Fs_vec[:,8] != 1))[0]   # identify triaxial load cases
    bi_idx = np.delete(np.arange(len(Fs_vec)), np.concatenate((ss_idx, tri_idx)))         # identify biaxial load cases

    # Define scaling parameters for different load cases because of the different stress magnitudes
    loadcase_idx = np.ones([len(Fs_vec), 1])
    loadcase_idx[ss_idx] = 1.
    loadcase_idx[bi_idx] = 20.
    loadcase_idx[tri_idx] = 1.
    loadcase_idx = loadcase_idx.reshape([len(Fs), num_loadcase*num_load_step, 1])

    adj = pickle.load(open(adj_pkl_file, "rb")).reshape([len(Fs)*max_node_num, max_node_num])
    nodes = pickle.load(open(node_pkl_file, "rb")).reshape([len(Fs)*max_node_num, 3])

    S = S_raw.reshape([len(Fs), num_loadcase*num_load_step, 9])
    Ws = W_raw
    truss_idx = np.arange(len(Fs))

    def detect_discontinuities(y, threshold = 0.1):
        discard = False
        for i in range(1, len(y)):
            if abs(y[i] - y[i-1]) > threshold and i > 10:
                discard = True
                break
        return discard

    def generate_datasets(truss_idx, load_idx_range, shuffle = False):

        F_data = np.zeros([len(truss_idx), len(load_idx_range)*num_load_step, 9])
        W_data = np.zeros([len(truss_idx), len(load_idx_range)*num_load_step, 1])
        S_data = np.zeros([len(truss_idx), len(load_idx_range)*num_load_step, 9])

        mask_data = np.zeros([len(truss_idx), len(load_idx_range)*num_load_step, 1])
        loadcase_idx_data = np.zeros([len(truss_idx), len(load_idx_range)*num_load_step, 1])
        adj_data = []
        nodes_data = []

        a_row, a_col = np.triu_indices(max_node_num)

        for uc_count, i in enumerate(truss_idx):
            adj_iterk = adj[i*max_node_num:(i+1)*max_node_num, :][a_row, a_col].flatten()
            nodes_iterk = nodes[i*max_node_num:(i+1)*max_node_num, :].flatten()[nodes_col]
            F_iterk = Fs[i, :, :]

            for load_count, j in enumerate(load_idx_range):
                j_idx = np.arange(j*num_load_step,(j+1)*num_load_step)

                if j not in [0,1,2]:
                    if detect_discontinuities(S[i, j_idx, 0]):
                        continue

                if np.linalg.norm(S[i, j_idx, :]) == 0:
                    mask_value = np.zeros([num_load_step,1])
                else:
                    mask_value = np.ones([num_load_step,1])

                # shuffle the load steps for each load case (optional)
                original_idx = j_idx
                if shuffle: 
                    shuffled_idx = np.random.permutation(original_idx)
                else:
                    shuffled_idx = original_idx.copy()

                F_data[uc_count, load_count*num_load_step:(load_count+1)*num_load_step, :] = F_iterk[shuffled_idx, :]
                S_data[uc_count, load_count*num_load_step:(load_count+1)*num_load_step, :] = S[i, shuffled_idx, :]
                W_data[uc_count, load_count*num_load_step:(load_count+1)*num_load_step, :] = Ws[i, shuffled_idx, :]
                mask_data[uc_count, load_count*num_load_step:(load_count+1)*num_load_step, :] = mask_value
                loadcase_idx_data[uc_count, load_count*num_load_step:(load_count+1)*num_load_step, :] = loadcase_idx[i, shuffled_idx, :]

            adj_data.append(adj_iterk)
            nodes_data.append(nodes_iterk)

        adj_data_tensor = torch.tensor(np.array(adj_data)).float()
        nodes_data_tensor = torch.tensor(np.array(nodes_data)).float()

        F_data_tensor = torch.tensor(np.array(F_data)).float()
        W_data_tensor = torch.tensor(np.array(W_data)).float()
        S_raw_data_tensor = torch.tensor(np.array(S_data)).float()
        S_data_tensor = S_raw_data_tensor
        mask_data_tensor = torch.tensor(np.array(mask_data)).float()
        loadcase_idx_data_tensor = torch.tensor(np.array(loadcase_idx_data)).float()

        if np.unique(mask_data_tensor.numpy()).shape[0] != 2:
            raise ValueError("Invalid dataset")

        dataset = TensorDataset(adj_data_tensor, nodes_data_tensor,
                                F_data_tensor,  W_data_tensor, S_data_tensor, mask_data_tensor, loadcase_idx_data_tensor)

        return dataset

    print("Building datasets ...")

    load_idx_range = np.arange(num_loadcase).tolist()
    dataset = generate_datasets(truss_idx, load_idx_range, shuffle = True)

    torch.save(dataset, f'{save_folder}/{num_graphs}_{dataset_type}_dataset.pt')
    print("Dataset size = ", len(dataset))
