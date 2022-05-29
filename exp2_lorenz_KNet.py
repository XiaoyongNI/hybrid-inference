import main_KNet as main
import numpy as np
import settings
import time
import utils.directory_utils as d_utils
from datasets.Extended_data import opt_q, r

print("1/r2 [dB]: ", 10 * np.log10(1/r**2))
print("1/q2 [dB]: ", 10 * np.log10(1/opt_q**2))

args = settings.get_settings()
args.exp_name = str(time.time())+'_lorenz_K%d' % args.taylor_K

args.batch_size = 1
args.init = 'meas_invariant'
args.gamma = 0.005
lr_base = args.lr


d_utils.init_folders(args.exp_name)
d_utils.copy_file('exp2_lorenz.py', 'logs/%s/%s' % (args.exp_name, 'exp2_lorenz.py'))
args.test_samples = 3000
print(args)


sweep_samples = np.array([2, 5, 10, 20,  50, 100, 200, 500, 1000, 2000, 5000, 10000]) * 10
epochs_arr = [1, 5, 30, 50, 60, 70, 70, 60, 60, 30, 25, 20]

sweep_K = [2, 5]
decimation = True # true for decimation case, false for DT case

def baseline():
    args.prior = False
    args.learned = False
    args.epochs = 0
    return main.main_lorenz_hybrid(args,decimation=decimation)


def only_prior(K=1, data=1000):
    args.K = 100
    args.tr_samples = 1
    args.val_samples = data
    args.learned = False
    args.prior = True
    args.epochs = 0
    opt_val_loss = 1e8
    for sigma in np.linspace(np.maximum(0.01,opt_q-1), opt_q+1, 5):       
        print("Sigma: %.4f, \t lamb: %.4f" % (sigma, r))
        val_loss, test_loss = main.main_lorenz_hybrid(args, sigma, lamb=r, K=K,decimation=decimation)
        if val_loss < opt_val_loss:
            opt_test = test_loss
            opt_val_loss = val_loss
            opt_sigma = sigma
            opt_lamb = r

    print("Sigma: %.4f, \t lamb: %.4f, \t Test_loss %.4f" % (opt_sigma, opt_lamb, opt_test))
    return opt_sigma, opt_test


def kalman(K=5, data=1000):
    args.tr_samples = 1
    args.val_samples = data
    args.learned = False
    args.prior = True
    args.epochs = 0
    opt_val_loss = 1e8
    for sigma in np.linspace(np.maximum(0.01,opt_q-1), opt_q+1, 5):         
        print("Sigma: %.4f, \t lamb: %.4f" % (sigma, r))
        val_loss, test_loss = main.main_lorenz_kalman(args, sigma=sigma, lamb=r, K=K,decimation=decimation)
        if val_loss < opt_val_loss:
            opt_test = test_loss
            opt_val_loss = val_loss
            opt_sigma = sigma
            opt_lamb = r

    print("Sigma: %.4f, \t lamb: %.4f, \t Test_loss %.4f" % (opt_sigma, opt_lamb, opt_test))
    return opt_sigma, opt_test




def hybrid(sigma, epochs, K=1, data=1000):
    args.K = 100
    args.tr_samples = int(data/2)
    args.val_samples = int(data/2)
    args.batch_size = 1
    args.prior = True
    args.learned = True
    args.epochs = epochs
    args.taylor_K=K
    val, test = main.main_lorenz_hybrid(args, sigma, lamb=r, K=K, dt = 0.02, plot_lorenz=True,decimation=decimation)
    return test


def only_learned(epochs, data=1000):
    args.K = 100
    args.tr_samples = int(data/2)
    args.val_samples = int(data/2)
    args.learned = True
    args.prior = False
    args.epochs = epochs
    val, test = main.main_lorenz_hybrid(args,decimation=decimation)
    best_val = 1e8
    test_error = 1e8
    if val < best_val:
        best_val = val
        test_error = test

    return test_error

if __name__ == '__main__':

    for K in sweep_K:
        key = 'K%d' % K
        results = {'prior': [], 'hybrid': [], 'sigma': [], 'n_samples': []}
        
        results_kalman = {'kalman_smoother': [], 'sigma_kalman': [], 'n_samples': []}
        if K > 0:
            for n_samples, epochs in zip(sweep_samples, epochs_arr):
                best_sigma, test_error = kalman(K, n_samples)
                results_kalman['kalman_smoother'].append(test_error)
                results_kalman['sigma_kalman'].append(best_sigma)
                results_kalman['n_samples'].append(n_samples)

                print('\nResults %s' % key)
                print(results_kalman)
                print('')
        print(results_kalman)
        
        if K > 0:
            args.lr = lr_base/2
            for n_samples, epochs in zip(sweep_samples, epochs_arr):
                print("\n######## \nOnly prior: start\n########\n")
                best_sigma, test_error = only_prior(K, n_samples)
                results['prior'].append(test_error)
                print("\n######## \nOnly prior: end\n########\n")

                print("\n######## \nHybrid: start\n########\n")
                test_error = hybrid(best_sigma, epochs, K=K, data=n_samples)
                results['hybrid'].append(test_error)
                results['sigma'].append(best_sigma)
                results['n_samples'].append(n_samples)
                print("\n######## \nHybrid: end\n########\n")

                print('\nResults %s' % key)
                print(results)
                print('')
        elif K == 0:
            args.lr = lr_base
            print("\n######## \nBaseline: start\n########\n")
            _, baseline_mse = baseline()
            print("\n######## \nBaseline: end\n########\n")
            for n_samples, epochs in zip(sweep_samples, epochs_arr):
                print("\n######## \nGNN: start\n########\n")
                test_error = only_learned(epochs, n_samples)
                results['prior'].append(baseline_mse)
                results['hybrid'].append(test_error)
                results['sigma'].append(-1)
                results['n_samples'].append(n_samples)
                print("\n######## \nGNN: end\n########\n")

                print('\nResults %s' % key)
                print(results)
                print('')
        d_utils.write_file('logs/%s/%s.txt' % (args.exp_name, key), str(results))


        