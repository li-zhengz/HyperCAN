import os
import math
import numpy as np

F_vec_dim = 9
max_node_num = 14
node_attr_dim = 3
num_graphs = 6000

num_data = num_graphs
a_row, a_col = np.triu_indices(max_node_num) # upper triangular indices
nodes_col = np.array([25, 26, 27, 29, 31, 32, 33, 35, 36, 37, 39, 40]) # other nodes are fixed

graph_t_dim = len(a_row) + len(nodes_col)
num_loadcase = 22
num_load_step = 100
random_seed = 123
valid_size = 0.05

target_dim = 50
label_dim = 50

raw_data_folder = '../data/dataset/train'
folder = '../data/'
scratch_dir = '../output/'

model_save_dir = f'{scratch_dir}/checkpoints'
results_save_dir = f'{scratch_dir}/results'

os.makedirs(folder, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(results_save_dir, exist_ok=True)

num_folds = 10

validation_index = 0
testing_index = 1

ICNN_n_input = 9
ICNN_n_output = 1

pi = 3.141592653589732

nodesInit = np.array([
    [0.,0.,0.],
    [1.,0.,0.],
    [1.,1.,0.],
    [0.,1.,0.],
    [0.,0.,1.],
    [1.,0.,1.],
    [1.,1.,1.],
    [0.,1.,1.],
    [0.,0.5,0.5],
    [0.5,1.,0.5],
    [1.,0.5,0.5],
    [0.5,0,0.5],
    [0.5,0.5,1],
    [0.5,0.5,0],
    # [0.5,0.5,0.5]  # center node
    ])