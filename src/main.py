from utils import *
from src.parameters import *
from icnn import *
import pickle
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
from sklearn.metrics import r2_score
import yaml

torch.cuda.empty_cache()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

torch.autograd.set_detect_anomaly(True)

with open("config.yaml", "r") as file:
    config_dict = yaml.safe_load(file)

config = dict_to_namespace(config_dict)

def train(graph_model, icnn_model, train_dataloader, validation_dataloader, epochs, checkpoint_dir, optimizer, criterion, validation_index, results_save_dir, derivative_calculation, weights):
    print('*' * 50)
    print('Training started')
    print('*' * 50)

    filename = f'{results_save_dir}/learning_Output.txt'
    output = open(filename, "w")
    print('Epoch, Training_time, Training_MSE, Validation_MSE',
          file=output, flush=True)

    F0 = torch.eye(3).float().to(device)
    F0 = F0.reshape([1, 9])

    for epoch in range(epochs):

        graph_model.train()
        icnn_model.train()

        total_macro_loss = []
        total_mse_loss = []
        total_nrmse_loss = []

        train_start_time = time.time()

        for adj, nodes, F, W, S, mask, loadcase_idx in train_dataloader:

            F_input_raw = tensor_to_variable(F).view(-1, F.shape[-1])  # (batch_size, num_loadcase * num_steps, 9)
            S_matrix_raw = tensor_to_variable(S).view(-1, S.shape[-1])
            W_label_raw = tensor_to_variable(W).view(-1, W.shape[-1])

            loadcase_idx_raw = tensor_to_variable(loadcase_idx).view(-1, loadcase_idx.shape[-1])
            loadcase_idx_raw[loadcase_idx_raw != 1.] = biaxial_scaling

            adj_raw = tensor_to_variable(adj)
            nodes_raw = tensor_to_variable(nodes)

            graph_t = torch.cat((adj_raw, nodes_raw), dim=1)
            graph_t = tensor_to_variable(graph_t).unsqueeze(1).repeat(1, F.shape[1], 1).view(-1, graph_t.shape[-1])
            
            # zero out the data that is not used
            mask = tensor_to_variable(mask).view(-1, mask.shape[-1]).bool().squeeze()
            graph_t = graph_t[mask, :]
            F_input = F_input_raw[mask, :]
            S_matrix = S_matrix_raw[mask, :]
            W_label = W_label_raw[mask, :]
            loadcase_idx = loadcase_idx_raw[mask, :]

            weights_out, skip_weights_out = graph_model(graph_t.float())

            # Get the deformation gradient tensor
            F11 = F_input[:, 0]; F12 = F_input[:, 1]; F13 = F_input[:, 2]
            F21 = F_input[:, 3]; F22 = F_input[:, 4]; F23 = F_input[:, 5]
            F31 = F_input[:, 6]; F32 = F_input[:, 7]; F33 = F_input[:, 8]

            # Compute the Cauchy-Green strain tensor
            C11, C12, C13, C21, C22, C23, C31, C32, C33 = computeCauchyGreenStrain(F11, F12, F13, F21, F22, F23, F31, F32, F33)
            
            # Compute the Green-Lagrange strain tensor
            E11, E12, E13, E21, E22, E23, E31, E32, E33 = computeGreenLagrangeStrain(C11, C12, C13, C21, C22, C23, C31, C32, C33)

            # Set the strain tensor to require gradient to calculate the derivative
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

            # Zero deformation gradient dummy data
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

            # Calculate the strain energy density
            W_NN = icnn_model(E_input.float(), weights_out, skip_weights_out)
            W0_NN = icnn_model(E0_input.float(), weights_out, skip_weights_out) 
            
            # Energy correction
            W_pred = W_NN - W0_NN

            # Calculate the second Piola-Kirchhoff stress tensor
            dW_dE = torch.autograd.grad(
                W_NN, E_input, grad_outputs=torch.ones_like(W_NN), create_graph=True)[0]
            
            # Stress at zero deformation
            dW0_dE0 = torch.autograd.grad(
                W0_NN, E0_input, grad_outputs=torch.ones_like(W0_NN), create_graph=True)[0]

            S_pred = dW_dE - dW0_dE0
            S_label = S_matrix.reshape(F_input.shape)

            # Calculate the loss
            stress_loss = torch.sqrt(mse_loss(S_pred.float()*loadcase_idx, S_label.float()*loadcase_idx))
            energy_loss = torch.sqrt(mse_loss(W_pred.float()*loadcase_idx, W_label.float()*loadcase_idx))
            loss = stress_loss*stress_loss_weight + energy_loss*energy_loss_weight

            total_macro_loss.append(macro_avg_err(S_pred, S_label).item())
            total_mse_loss.append(torch.sqrt(
                mse_loss(S_pred, S_label)).item())
            total_nrmse_loss.append(nrmse(S_pred.cpu().detach().numpy(), S_label.cpu().detach().numpy()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_end_time = time.time()
        training_loss_epoch = np.mean(total_mse_loss)
        training_nrmse_epoch = np.mean(total_nrmse_loss)

        # Save the model if the validation loss is the best so far
        if epoch == 0:
            write_results = True
            torch.save(graph_model, f'{model_save_dir}/graph_checkpoint.pth')
            torch.save(icnn_model, f'{model_save_dir}/icnn_checkpoint.pth')
            best_loss = training_loss_epoch

        else:
            if training_loss_epoch < best_loss:
                write_results = True
                torch.save(graph_model, f'{model_save_dir}/graph_checkpoint.pth')
                torch.save(icnn_model, f'{model_save_dir}/icnn_checkpoint.pth')
                best_loss = training_loss_epoch
                print(f'Saving model at epoch {epoch} in {model_save_dir}')
            else:
                write_results = False
                
        validation_nrmse_epoch, validation_loss_epoch, S_r2_score = test(
            graph_model, icnn_model, validation_dataloader, 'Validation', write_results, criterion, validation_index, results_save_dir, derivative_calculation)

        print('%d, %.3f, %e, %e' % (epoch, train_end_time-train_start_time,
              training_loss_epoch, validation_loss_epoch), file=output, flush=True)
        torch.cuda.empty_cache()


def test(graph_model, icnn_model, test_dataloader, test_val_tr, printcond, criterion, running_index, results_save_dir, derivative_calculation):
    graph_model.eval()
    icnn_model.eval()

    y_label_list, y_pred_list, total_loss = [], [], 0
    F_list, W_label_list = [], []
    W_pred_list = []

    F0 = torch.eye(3).float().to(device)
    F0 = F0.reshape(1, 9)
    total_mse = []
    total_nrmse = []

    for adj, nodes, F, W, S, mask, loadcase_idx in train_dataloader:

        F_input_raw = tensor_to_variable(F).view(-1, F.shape[-1])  # (batch_size, num_loadcase, 9)
        S_matrix_raw = tensor_to_variable(S).view(-1, S.shape[-1])
        W_label_raw = tensor_to_variable(W).view(-1, W.shape[-1])

        loadcase_idx_raw = tensor_to_variable(loadcase_idx).view(-1, loadcase_idx.shape[-1])
        loadcase_idx_raw[loadcase_idx_raw != 1.] = biaxial_scaling

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

        C11, C12, C13, C21, C22, C23, C31, C32, C33 = computeCauchyGreenStrain(F11, F12, F13, F21, F22, F23, F31, F32, F33)

        weights_out, skip_weights_out = graph_model(
            graph_t.float())

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

        S_label_unnorm = S_label
        S_pred_unnorm = S_pred
        W_label_unnorm = W_label
        W_pred_unnorm = W_pred

        y_pred_list.extend(variable_to_numpy(S_pred_unnorm))
        y_label_list.extend(variable_to_numpy(S_label_unnorm))
        F_list.extend(variable_to_numpy(F_input))
        W_label_list.extend(variable_to_numpy(W_label_unnorm))
        W_pred_list.extend(variable_to_numpy(W_pred_unnorm))

        total_nrmse.append(nrmse(variable_to_numpy(S_pred_unnorm), variable_to_numpy(S_label_unnorm)))
        total_mse.append(torch.sqrt(mse_loss(S_pred, S_label)).item())

    y_label_list = np.array(y_label_list)
    y_pred_list = np.array(y_pred_list)
    F_list = np.array(F_list)
    W_label_list = np.array(W_label_list)
    W_pred_list = np.array(W_pred_list)
    total_loss = r2_score(y_label_list, y_pred_list)
    total_nrmse_epoch = np.mean(total_nrmse)
    total_mse_epoch = np.mean(total_mse)

    S11_r2_score = r2_score(y_label_list[:, 0], y_pred_list[:, 0])
    S22_r2_score = r2_score(y_label_list[:, 4], y_pred_list[:, 4])
    S33_r2_score = r2_score(y_label_list[:, 8], y_pred_list[:, 8])
    S_r2_score = [S11_r2_score, S22_r2_score, S33_r2_score]

    if printcond:
        y_label_list_save = y_label_list
        y_pred_list_save = y_pred_list

        filename = f'{results_save_dir}/{test_val_tr}_S_label.pkl'
        pickle.dump(y_label_list_save, open(filename, "wb"))

        filename = f'{results_save_dir}/{test_val_tr}_S_pred.pkl'
        pickle.dump(y_pred_list_save, open(filename, "wb"))

        filename = f'{results_save_dir}/{test_val_tr}_F_label.pkl'
        pickle.dump(F_list, open(filename, "wb"))

        filename = f'{results_save_dir}/{test_val_tr}_W_label.pkl'
        pickle.dump(W_label_list, open(filename, "wb"))

        filename = f'{results_save_dir}/{test_val_tr}_W_pred.pkl'
        pickle.dump(W_pred_list, open(filename, "wb"))

    return total_nrmse_epoch, total_mse_epoch, S_r2_score


if __name__ == '__main__':

    os.environ['PYTHONHASHargs.seed'] = str(random_seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True

    epochs = config.epochs
    learning_rate = config.learning_rate
    batch_size = config.batch_size

    mlp_output_dim = config.mlp_output_dim
    t_n_input = mlp_output_dim
    icnn_activation_func = config.icnn_activation_func
    c_hidden_dim = config.c_hidden_dim
    derivative_calculation = config.derivative_calculation
    learner_dim = config.learner_dim
    diagonal_weight = config.diagonal_weight
    non_diagonal_weight = config.non_diagonal_weight
    layer_type = config.layer_type
    scaling_sftpSq = config.scaling_sftpSq
    fully_learner_dim = config.fully_learner_dim
    skip_learner_dim = config.skip_learner_dim
    activation_func = config.activation_func

    biaxial_scaling = config.biaxial_scaling

    energy_loss_weight = config.energy_loss_weight
    stress_loss_weight = config.stress_loss_weight

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mse_loss = nn.MSELoss(reduction='mean')
    criterion = weighted_mse_loss

    in_optim = "Adam"
    weights = torch.ones([9])
    if diagonal_weight[0]:
        weights[0] *= diagonal_weight[1]
        weights[4] *= diagonal_weight[1]
        weights[8] *= diagonal_weight[1]
    if non_diagonal_weight[0]:
        weights[1] *= non_diagonal_weight[1]
        weights[2] *= non_diagonal_weight[1]
        weights[3] *= non_diagonal_weight[1]
        weights[5] *= non_diagonal_weight[1]
        weights[6] *= non_diagonal_weight[1]
        weights[7] *= non_diagonal_weight[1]
    weights = weights.float().to(device)
    icnn_model = constitutiveNN(c_hidden_dim, icnn_activation_func, layer_type, scaling_sftpSq)

    weights_output_dim = []
    skip_weights_output_dim = []
    for name, w in icnn_model.named_parameters():
        if 'skip' in name:
            skip_weights_output_dim.extend([w.shape[0]*w.shape[1]])
        else:
            weights_output_dim.extend([w.shape[0]*w.shape[1]])

    graph_model = multi_task_learner(graph_t_dim, learner_dim, fully_learner_dim, skip_learner_dim, weights_output_dim, skip_weights_output_dim, activation_func)
    graph_model.initialize_kaiming_weights()
    print(graph_model)
    print(device)
    
    graph_model.float().to(device)
    icnn_model.float().to(device)

    if in_optim == "Adam":
        optimizer = optim.Adam(graph_model.parameters(), lr=learning_rate)

    train_start_time = time.time()
    train_load_dataset = torch.load(folder+'/dataset/'+str(num_graphs)+'_train_dataset.pt')

    num_train = len(train_load_dataset)
    split = int(np.floor(valid_size * num_train))
    train_dataset, test_dataset = random_split(
        dataset=train_load_dataset, lengths=[num_train - split, split])
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=len(
        test_dataset), shuffle=False, pin_memory=True)

    num_iters = len(train_dataloader)
    train(graph_model, icnn_model, train_dataloader, test_dataloader, epochs, model_save_dir,
          optimizer, criterion, validation_index, results_save_dir, derivative_calculation, weights)
    train_end_time = time.time()

    torch.save(graph_model, f'{model_save_dir}/graph_checkpoint.pth')
    torch.save(icnn_model, f'{model_save_dir}/icnn_checkpoint.pth')

    train_rel, train_mse, train_r2_score = test(graph_model, icnn_model, test_dataloader, 'Training',
                                True, criterion, validation_index, results_save_dir, derivative_calculation)
    validation_rel, validation_mse, validation_r2_score = test(graph_model, icnn_model, test_dataloader,
                                          'Validation', True, criterion, validation_index, results_save_dir, derivative_calculation)
    test_rel, test_mse, test_r2_score = test(graph_model, icnn_model, test_dataloader, 'Test',
                              True, criterion, testing_index, results_save_dir, derivative_calculation)

    print('--------------------')
    print("validation_index : {}".format(validation_index))
    print("testing_index : {}".format(testing_index))
    print("Train Relative Error: {:.3f}%".format(100 * train_rel))
    print("Validation Relative Error: {:.3f}%".format(100 * validation_rel))
    print("Test Relative Error: {:.3f}%".format(100 * test_rel))
    print("Train MSE : {}".format(train_mse))
    print("Validation MSE : {}".format(validation_mse))
    print("Test MSE: {}".format(test_mse))
