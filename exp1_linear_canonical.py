import settings
import main_KNet
import numpy as np
import utils.directory_utils as d_utils
import time
from datasets.Extended_data import q, r

print("1/r2 [dB]: ", 10 * np.log10(1/r**2))
print("1/q2 [dB]: ", 10 * np.log10(1/q**2))

args = settings.get_settings()
args.exp_name = str(time.time())+'_linear'
d_utils.init_folders(args.exp_name)

args.batch_size = 1
args.gamma = 0.003
args.test_samples = 10*1000
args.init = 'meas_invariant'
args.lr = 1e-3
args.K = 100
args.prior = True
args.learned = True
best_sigma = q
   
print(args)
lr_base = float(args.lr)
sweep_samples = np.array([10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 10000]) * 10
epochs_arr = [40, 40, 60, 50, 50, 50, 40, 50, 50, 40, 25]
lr_arr = [lr_base/10, lr_base/5, lr_base/2] + [lr_base] * (len(epochs_arr) - 5)


if len(sweep_samples) != len(epochs_arr):
    raise Exception('Arrays "sweep_samples" and "epochs_arr" do not match.')




def baseline():
    print("\n######## \nBaseline: start\n########\n")
    args.prior = False
    args.learned = False
    args.epochs = 1
    _, test = main_KNet.synthetic_hybrid(args)
    print("\n######## \nBaseline: end\n########\n")
    return test

# def kalman(data):
#     best_val = 1e8
#     best_sigma = 0.15, 0.15
#     test_error = 1e8
#     args.tr_samples = int(data/2)
#     args.val_samples = data - args.tr_samples
#     # Best: sigma --> 0.063, lamb --> 0.4833
#     for sigma in np.linspace(q-0.5, 0.5, q+8):
#         print("Kalman Smoother sigma %.4f" % sigma)
#         val, test = main_KNet.main_synhtetic_kalman(args, sigma, val_on_train=True, optimal=False)
#         if val < best_val:
#             best_val = val
#             best_sigma = sigma
#             test_error = test
#     print("Kalman Smoother error: %.4f" % test_error)
#     return best_sigma, test_error

def kalman_optimal():
    _, test = main_KNet.main_synhtetic_kalman(args, sigma=q, lamb=r, val_on_train=False, optimal=True)
    return test



def hybrid(sigma, data=100, epochs=1,test_time=False):
    args.tr_samples = int(data*0.5)
    args.val_samples = int(data - args.tr_samples)
    args.epochs = epochs
    if test_time:
        mse = main_KNet.synthetic_hybrid(args, sigma=sigma,lamb=r,test_time=True)
    else:
        mse = main_KNet.synthetic_hybrid(args, sigma=sigma,lamb=r,test_time=False)
    return mse




if __name__ == '__main__':
    ## Baseline ##
    # base = baseline() ## 'baseline': base,
    results = {'hybrid test': [], 'hybrid val': [],'kalman_optimal':[]}

    #results = {'prior': [], 'learned': [], 'hybrid': [], 'sigma': [], 'lamb':[], 'sigma_kalman': [], 'n_samples': [], 'kalman':[], 'kalman_optimal':[]}
     ## Kalman Smoother Optimal ##
    print("\n######## \nKalman Optimal: start\n########\n")
    test_error = kalman_optimal()
    results['kalman_optimal'].append(test_error)
    print("\n######## \nKalman Optimal: end\n########\n")

    for n_samples, epochs, lr in zip(sweep_samples, epochs_arr, lr_arr):
        args.lr = lr
        print("### Linear experiment n_samples: %d \t epochs: %d \t Learning Rate: %f" % (n_samples, epochs, lr))

        ## Kalman Smoother ##
        # print("\n######## \nKalman Smoother: start\n########\n")
        # sigma, test_error = kalman(n_samples)
        # results['kalman'].append(test_error)
        # results['sigma_kalman'].append(sigma)
        # print("\n######## \nKalman Smoother: end\n########\n")

       


        ## Hybrid ##
        print("\n######## \nHybrid: start\n########\n")
        val_error = hybrid(best_sigma, data=n_samples, epochs=int(epochs),test_time=False)
        #test_error = hybrid(0.425, data=n_samples, epochs=int(epochs))
        val_error_dB = 10 * np.log10(val_error)
        results['hybrid val'].append(val_error_dB)
        print("\n######## \nHybrid: end\n########\n")

        # Meta-informations
        print('\nResults %s \n' % str(results))
        print('')
    
    print("\n######## \nTest Hybrid: start\n########\n")
    test_error = hybrid(best_sigma, data=n_samples, epochs=1,test_time=True)
    test_error_dB = 10 * np.log10(test_error)
    results['hybrid test'].append(test_error_dB)
    print('\nResults %s'% str(results))
    print(results)
    print('')

    d_utils.write_file('logs/%s/log.txt' % (args.exp_name), str(results))


