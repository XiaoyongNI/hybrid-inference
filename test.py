from utils import generic_utils as g_utils
import torch
import evaluation as eval
import torch.nn.functional as F
import losses
from datasets import nclt
from datasets import synthetic
from datasets import lorenz_KNet as lorenz
import numpy as np
from utils import generic_utils as g_utils
import time
from datasets.Extended_data import wandb_switch
if wandb_switch:
    import wandb

def test_kalman(args, model, test_loader, plots=False, nclt_ds=False):
    test_loss = 0

    for state, meas, x_0, P_0 in test_loader:
        batch_size = state.size()[0]
        for i in range(batch_size):
            state_np = state.numpy()[i, :, :]
            meas_np = meas.numpy()[i, :, :]
            x_0_np = x_0.numpy()[i, :]
            P_0_np = P_0.numpy()[i, :]
            # g_utils.plot_trajectory(state, meas)
            est_state, est_cov = model.forward(meas_np, x_0_np, P_0_np)



            if nclt_ds:
                if plots:
                    nclt.plot_trajecotry([state_np, meas_np])
                    nclt.plot_trajecotry([state_np, g_utils.state2position(est_state)])
                sample_loss = eval.mse(state_np, g_utils.state2position(est_state))
            else:
                if plots:
                    g_utils.plot_prediction(state_np, meas_np, est_state, est_cov)
                sample_loss = eval.mse(g_utils.state2position(state_np), g_utils.state2position(est_state))
            test_loss += sample_loss

    test_loss /= len(test_loader.dataset)
    print('%s set: Average loss: {:.4f}, Num samples: {}\n'.format(test_loader.dataset.partition,
        test_loss, len(test_loader.dataset)))
    return test_loss


def test_kalman_nclt(model, test_loader, plots=False,test_time=False):
    test_loss = 0
    j = 0
    sample_loss = torch.empty([len(test_loader.dataset)])
    start = time.time()
    for _, state, meas, x_0, P_0, _ in test_loader:
        batch_size = state.size()[0]
        for i in range(batch_size):
            state_np = state.numpy()[i, :, :]
            meas_np = meas.numpy()[i, :, :]
            x_0_np = x_0.numpy()[i, :]
            P_0_np = P_0.numpy()[i, :]
            # g_utils.plot_trajectory(state, meas)
            est_state, est_cov = model.forward(meas_np, x_0_np, P_0_np)

            if plots:
                synthetic.plot_trajecotry([state_np, g_utils.state2position(est_state)])
            sample_loss[j] = torch.from_numpy(np.asarray(eval.mse(state_np, g_utils.state2position(est_state), normalize=False)))
            test_loss += sample_loss[j]
            sample_loss[j] /= state.size()[1]##average over traj length T  
            j += 1
    end = time.time()
    t = end - start         
    test_loss /= test_loader.dataset.total_len()
    print('{} set: Average loss: {:.4f}, Num samples: {}\n'.format(test_loader.dataset.partition,
        test_loss, len(test_loader.dataset)))
    
    ### MSE loss in dB
    test_loss_dB = 10 * torch.log10(test_loss)
    print("MSE LOSS:", test_loss_dB, "[dB]")
    # Optional: record to wandb
    if wandb_switch:
        if(test_time):
            wandb.summary['test loss'] = test_loss_dB
        else:
            wandb.log({"loss": test_loss_dB})

    ### std Standard deviation
    MSE_test_linear_std = torch.std(sample_loss, unbiased=True)
    # Confidence interval
    test_std_dB = 10 * torch.log10(MSE_test_linear_std + test_loss) - test_loss_dB
    print("MSE LOSS std:", test_std_dB, "[dB]")

    # Print Run Time
    print("Inference Time:", t)

    return test_loss


def test_kalman_lorenz(args, model, test_loader, plots=False):
    test_loss = 0
    j = 0
    sample_loss = torch.empty([len(test_loader.dataset)])
    start = time.time()
    for _, state, meas, x_0, P_0, _ in test_loader:
        batch_size = state.size()[0]
        for i in range(batch_size):
            state_np = state.numpy()[i, :, :]
            meas_np = meas.numpy()[i, :, :]
            x_0_np = x_0.numpy()[i, :]
            P_0_np = P_0.numpy()[i, :]
            est_state = model.forward(meas_np)

            sample_loss[j] = torch.from_numpy(np.asarray(eval.mse(state_np, est_state, normalize=False)))
            test_loss += sample_loss[j]
            sample_loss[j] /= state.size()[1]##average over traj length T
            j += 1
    test_loss /= test_loader.dataset.total_len()
    end = time.time()
    t = end - start   
    if plots:
        lorenz.plot_trajectory(args, est_state, test_loss)
    print('{} set: Average loss: {:.4f}, Num samples: {}\n'.format(test_loader.dataset.partition,
        test_loss, len(test_loader.dataset)))

    test_loss_dB = 10 * torch.log10(test_loss)
    print("MSE LOSS:", test_loss_dB, "[dB]")

    # Standard deviation
    MSE_test_linear_std = torch.std(sample_loss, unbiased=True)

    # Confidence interval
    test_std_dB = 10 * torch.log10(MSE_test_linear_std + test_loss) - test_loss_dB
    print("MSE LOSS std:", test_std_dB, "[dB]")

    # Print Run Time
    print("Inference Time:", t)
    return test_loss


def test_gnn_kalman(args, net, device, loader, plots=False, plot_lorenz=False,test_time=False):
    net.eval()
    test_loss = 0
    test_mse = 0
    j = 0
    sample_loss = torch.empty([len(loader.dataset)])
    T = loader.dataset.total_len()/len(loader.dataset)
    start = time.time()
    with torch.no_grad():
        for batch_idx, (ts, position, meas, x0, P0, operators) in enumerate(loader):
            position, meas, x0 = position.to(device), meas.to(device), x0.to(device)
            operators = g_utils.operators2device(operators, device)
            outputs = net([operators, meas], x0, args.K, ts=ts)
            sample_loss[j] = F.mse_loss(outputs[-1], position) * meas.size()[0] * meas.size()[1]
            test_mse = test_mse + sample_loss[j]
            sample_loss[j] /= T ##average over traj length T
            j += 1
            test_loss += losses.mse_arr_loss(outputs, position) * meas.size()[0] * meas.size()[1]
        test_mse /= loader.dataset.total_len() + 1e-10
        test_loss /= loader.dataset.total_len() + 1e-10
    
    end = time.time()
    t = end - start

    if plot_lorenz:
        lorenz.plot_trajectory(args, outputs[-1][0].cpu().numpy(), test_mse)

    print('\t{} set: Loss: {:.4f}, MSE: {:.4f}, Len {}'.format(loader.dataset.partition,
        test_loss, test_mse, len(loader.dataset)))
    
    # try:
    #   test_mse_dB = 10 * np.log10(test_mse.cpu().numpy())
    # except:
    #   test_mse_dB = 10 * np.log10(test_mse)
    test_mse_dB = 10 * torch.log10(test_mse)
    print("MSE LOSS:", test_mse_dB, "[dB]")
    # Optiional: record to wandb
    if wandb_switch:
        if(test_time):
            wandb.summary['test loss'] = test_mse_dB
        else:
            wandb.log({"loss": test_mse_dB})

    # Standard deviation
    MSE_test_linear_std = torch.std(sample_loss, unbiased=True)

    # Confidence interval
    test_std_dB = 10 * torch.log10(MSE_test_linear_std + test_mse) - test_mse_dB
    print("MSE LOSS std:", test_std_dB, "[dB]")

    # Print Run Time
    print("Inference Time:", t)
    return test_mse
