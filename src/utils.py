from src.__importList__ import *
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from src.parameters import *
import math
from tqdm import trange
import shutil
from types import SimpleNamespace

pi = 3.141592653589732
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    ])

def dict_to_namespace(d):
    for key, value in d.items():
        if key == 'values' and isinstance(value, list) and len(value) > 0:
            return value[0]  
        elif isinstance(value, dict):
            d[key] = dict_to_namespace(value)
    return SimpleNamespace(**d)

def calculate_nrmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2)) / (np.max(y_true) - np.min(y_true))

def weighted_loss_with_columns(output, target, weights):
    # Compute the squared error for each column (feature)
    weights = weights.to(device)
    squared_error = (output - target) ** 2
    weighted_error = squared_error * weights
    loss = torch.mean(weighted_error)
    return loss

def weighted_mse_loss(input, target, weight):
    return (weight * (input - target) ** 2).mean()

def tensor_to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x.float())
    # return Variable(x)

def variable_to_numpy(x):
    if torch.cuda.is_available():
        x = x.cpu()
    x = x.data.numpy()
    return x

def print_preds(y_label_list, y_pred_list, test_or_tr):
    length, w = np.shape(y_label_list)
    print()
    print('{} Set Predictions: '.format(test_or_tr))
    for i in range(0, length):
        print('True:{}, Predicted: {}'.format(y_label_list[i], y_pred_list[i]))

def mse(Y_prime, Y):
    return np.mean((Y_prime - Y) ** 2)


def macro_avg_err(Y_prime, Y):
    if type(Y_prime) is np.ndarray:
        return np.sum(np.abs(Y - Y_prime)) / np.sum(np.abs(Y))
    return torch.sum(torch.abs(Y - Y_prime)) / torch.sum(torch.abs(Y))


def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2)


def compute1stPKTtensor(W, F11, F12, F13, F21, F22, F23, F31, F32, F33):
    '''
    Compute the first Piola-Kirchhoff stress tensor (1st PK tensor).
    
    The first Piola-Kirchhoff stress tensor relates the deformation gradient to the stress 
    in the material and is computed by differentiating the strain energy function `W` 
    with respect to each component of the deformation gradient tensor `F`. 

    Parameters:
    W    : Strain energy (scalar tensor that depends on the deformation gradient)
    F11, F12, F13, F21, F22, F23, F31, F32, F33 : Components of the deformation gradient tensor F (3x3)

    Returns:
    P    : The first Piola-Kirchhoff stress tensor (3x3 for each sample)
         - Tensor with the same number of samples as the input deformation gradients.
    '''

    grad_shape = torch.ones(F11.shape[0], 1).to(device)
    dW_dF11 = torch.autograd.grad(W, F11, grad_shape, create_graph=True)[
        0].unsqueeze(-1)
    dW_dF12 = torch.autograd.grad(W, F12, grad_shape, create_graph=True)[
        0].unsqueeze(-1)
    dW_dF13 = torch.autograd.grad(W, F13, grad_shape, create_graph=True)[
        0].unsqueeze(-1)
    dW_dF21 = torch.autograd.grad(W, F21, grad_shape, create_graph=True)[
        0].unsqueeze(-1)
    dW_dF22 = torch.autograd.grad(W, F22, grad_shape, create_graph=True)[
        0].unsqueeze(-1)
    dW_dF23 = torch.autograd.grad(W, F23, grad_shape, create_graph=True)[
        0].unsqueeze(-1)
    dW_dF31 = torch.autograd.grad(W, F31, grad_shape, create_graph=True)[
        0].unsqueeze(-1)
    dW_dF32 = torch.autograd.grad(W, F32, grad_shape, create_graph=True)[
        0].unsqueeze(-1)
    dW_dF33 = torch.autograd.grad(W, F33, grad_shape, create_graph=True)[
        0].unsqueeze(-1)

    P_vec = torch.cat((dW_dF11, dW_dF12, dW_dF13, dW_dF21,
                      dW_dF22, dW_dF23, dW_dF31, dW_dF32, dW_dF33), dim=1)
    P = P_vec.reshape((len(F11), 3, 3))

    return P

def computeCauchyGreenStrain(F11, F12, F13, F21, F22, F23, F31, F32, F33):
    '''
    Compute the Cauchy-Green strain tensor (C = F^T F).

    Parameters:
    F11, F12, F13, F21, F22, F23, F31, F32, F33 : Components of the deformation gradient tensor F (3x3)

    Returns:
    C11, C12, C13, C21, C22, C23, C31, C32, C33 : Components of the right Cauchy-Green strain tensor (3x3)
    
    '''

    C11 = F11 ** 2 + F21 ** 2 + F31 ** 2
    C12 = F11 * F12 + F21 * F22 + F31 * F32
    C13 = F11 * F13 + F21 * F23 + F31 * F33
    C21 = F12 * F11 + F22 * F21 + F32 * F31
    C22 = F12 ** 2 + F22 ** 2 + F32 ** 2
    C23 = F12 * F13 + F22 * F23 + F32 * F33
    C31 = F13 * F11 + F23 * F21 + F33 * F31
    C32 = F13 * F12 + F23 * F22 + F33 * F32
    C33 = F13 ** 2 + F23 ** 2 + F33 ** 2
    return C11, C12, C13, C21, C22, C23, C31, C32, C33

def computeGreenLagrangeStrain(C11, C12, C13, C21, C22, C23, C31, C32, C33):
    '''
    Compute the Green-Lagrange strain tensor (E = 1/2 * (C - I)).

    Parameters:
    C11, C12, C13, C21, C22, C23, C31, C32, C33 : Components of the right Cauchy-Green strain tensor (3x3)

    Returns:
    E11, E12, E13, E21, E22, E23, E31, E32, E33 : Components of the Green-Lagrange strain tensor (3x3)
    
    '''
    E11 = 1/2.*(C11-1)
    E12 = 1/2.*C12
    E13 = 1/2.*C13
    E21 = 1/2.*C21
    E22 = 1/2.*(C22-1)
    E23 = 1/2.*C23
    E31 = 1/2.*C31
    E32 = 1/2.*C32
    E33 = 1/2.*(C33-1)

    return E11, E12, E13, E21, E22, E23, E31, E32, E33

def delete_all_folders(directory):
    # Iterate over all items in the directory
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        # Check if the item is a directory
        if os.path.isdir(item_path):
            # Delete the directory and its contents recursively
            shutil.rmtree(item_path)

def nrmse(predictions, targets):
    """
    Compute the normalized root mean square error (NRMSE).

    Parameters:
    predictions (array-like): Predicted values
    targets (array-like): True values

    Returns:
    float: The NRMSE value
    """
    predictions = np.array(predictions)
    targets = np.array(targets)

    nrmse = np.sqrt(np.mean((predictions - targets) ** 2)) / (np.max(targets) - np.min(targets))
    # nrmse = np.sqrt(np.mean((predictions - targets) ** 2) / np.mean(targets ** 2))
    return nrmse