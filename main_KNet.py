# from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from models import kalman, gnn
import settings
from datasets import lorenz_KNet, synthetic_KNet, nclt
import test
import losses
from datasets.dataloader import DataLoader
from utils import generic_utils as g_utils
from datasets.Extended_data import m1_0, m2_0, CA_m1_0, CA_m2_0, CV_m1_0, CV_m2_0,lor_T, lor_T_test,CV_model,Train_Loss_OnlyP

def train_hybrid(args, net, device, train_loader, optimizer, epoch):
    net.train()
    stepsxsample = 1.0 * train_loader.dataset.total_len() / (len(train_loader.dataset) + 1e-12)
    for batch_idx, (ts, position, meas, x0, P0, operators) in enumerate(train_loader):
        position, meas, x0 = position.to(device), meas.to(device), x0.to(device)
        operators = g_utils.operators2device(operators, device)
        optimizer.zero_grad()
        outputs = net([operators, meas], x0, args.K, ts=ts)
        mse = F.mse_loss(outputs[-1], position)
        loss = losses.mse_arr_loss(outputs, position)
        if meas.size(0) == 1:
            loss = loss * meas.size(1) / stepsxsample
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tMSE: {:.6f}'.format(
                epoch, batch_idx * len(meas), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), mse.item()))
            
def train_hybrid_CA(args, net, device, train_loader, optimizer, epoch):
    net.train()
    stepsxsample = 1.0 * train_loader.dataset.total_len() / (len(train_loader.dataset) + 1e-12)
    for batch_idx, (ts, position, meas, x0, P0, operators) in enumerate(train_loader):
        position, meas, x0 = position.to(device), meas.to(device), x0.to(device)
        operators = g_utils.operators2device(operators, device)
        optimizer.zero_grad()
        outputs = net([operators, meas], x0, args.K, ts=ts)
        ### eliminate double dim
        if train_loader.dataset.equations == 'CA':
            for i in range(len(outputs)):
                if CV_model:
                    outputs[i] = outputs[i][:,:,0:2]
                else:
                    outputs[i] = outputs[i][:,:,0:3]

        if Train_Loss_OnlyP: 
            mse = F.mse_loss(outputs[-1][:,:,0], position[:,:,0])
            loss = losses.mse_arr_loss(outputs, position,Test_Loss_OnlyP=True)
        else:
            mse = F.mse_loss(outputs[-1], position)
            loss = losses.mse_arr_loss(outputs, position)
        if meas.size(0) == 1:
            loss = loss * meas.size(1) / stepsxsample
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tMSE: {:.6f}'.format(
                epoch, batch_idx * len(meas), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), mse.item()))


def adjust_learning_rate(optimizer, lr, epoch):
    new_lr = lr * (0.5 ** (epoch // (args.epochs/5)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def CA_hybrid(args, val_on_train=False, load = True,test_time=False):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    args.device = device
    print("working on device %s" % device)
    
    dataset_test = synthetic_KNet.SYNTHETIC(partition='test', tr_tt=args.tr_samples, val_tt=args.val_samples, test_tt=args.test_samples,
                                         equations="CA", gnn_format=True, load = load)
    test_loader = DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False)

    dataset_train = synthetic_KNet.SYNTHETIC(partition='train', tr_tt=args.tr_samples, val_tt=args.val_samples, test_tt=args.test_samples,
                                        equations="CA", gnn_format=True, load = load)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)

    dataset_val = synthetic_KNet.SYNTHETIC(partition='val', tr_tt=args.tr_samples, val_tt=args.val_samples, test_tt=args.test_samples,
                                      equations="CA", gnn_format=True, load = load)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)

    # (A, H, Q, R) = synthetic.create_model_parameters_v(s2_x=sigma ** 2, s2_y=sigma ** 2, lambda2=lamb ** 2)
    (A, H, Q, R) = synthetic_KNet.create_model_parameters_CA()
    
    if test_time:### testing
        print("Start testing")
        try:
            net = torch.load(args.path_results+'best-model.pt', map_location=device)
        except:
            print("Error loading the trained model!!!")
        test_mse = test.test_gnn_kalman_CA(args, net, device, test_loader, plots=False,test_time=True)

        print("Test loss: %.4f" % (test_mse))
        return test_mse.item()
    
    else:### training  
        try:        
            net = torch.load(args.path_results+'best-model.pt', map_location=device)
            print("Load network from previous training")
            NumofParameter = sum(p.numel() for p in net.parameters() if p.requires_grad)
            print("Number of parameters for Hybrid model: ",NumofParameter)  
        except:
            net = gnn.GNN_Kalman(args, A, H, Q, R, dataset_train.x0,  dataset_train.P0, nf=args.nf,
                                prior=args.prior,  learned=args.learned, init=args.init, gamma=args.gamma).to(device)
            print("Initialize Network")
            torch.save(net, args.path_results + 'best-model.pt')          
            NumofParameter = sum(p.numel() for p in net.parameters() if p.requires_grad)
            print("Number of parameters for Hybrid model: ",NumofParameter)    
   
        #print(net)
        optimizer = optim.Adam(net.parameters(), lr=args.lr)

        min_val = test.test_gnn_kalman_CA(args, net, device, val_loader)

        for epoch in range(1, args.epochs + 1):
            #adjust_learning_rate(optimizer, args.lr, epoch)
            train_hybrid_CA(args, net, device, train_loader, optimizer, epoch)

            val_mse = test.test_gnn_kalman_CA(args, net, device, val_loader)
            
            if val_on_train:
                train_mse = test.test_gnn_kalman_CA(args, net, device, train_loader)
                val_mse = (val_mse * dataset_val.total_len() + train_mse * dataset_train.total_len())/(dataset_val.total_len() + dataset_train.total_len())

            if val_mse < min_val:
                min_val = val_mse
                ### save best model on validation set
                torch.save(net, args.path_results + 'best-model.pt')
        return min_val.item()


def main_CA_kalman(args, val_on_train=False, optimal=False, load = True):

    if val_on_train:
        dataset_train = synthetic_KNet.SYNTHETIC(partition='train', tr_tt=args.tr_samples, val_tt=args.val_samples,
                                            test_tt=args.test_samples,
                                            equations="CA", gnn_format=True, load = load)
        train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False)

    dataset_val = synthetic_KNet.SYNTHETIC(partition='val', tr_tt=args.tr_samples, val_tt=args.val_samples, test_tt=args.test_samples,
                                      equations="CA", gnn_format=True, load = load)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)

    dataset_test = synthetic_KNet.SYNTHETIC(partition='test', tr_tt=args.tr_samples, val_tt=args.val_samples, test_tt=args.test_samples,
                                       equations="CA", gnn_format=True, load = load)
    test_loader = DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False)

    (A, H, Q, R) = synthetic_KNet.create_model_parameters_CA()
    if CV_model:
        ks_v = kalman.KalmanSmoother(A, H, Q, R, CV_m1_0, CV_m2_0)
        print('Testing Kalman Smoother CV case')
    else:
        ks_v = kalman.KalmanSmoother(A, H, Q, R, CA_m1_0, CA_m2_0)   
        print('Testing Kalman Smoother CA case')

    # val_loss = test.test_kalman_nclt(ks_v, val_loader, plots=False)
    test_loss = test.test_kalman_nclt(ks_v, test_loader, plots=False)

    if val_on_train:
        train_loss = test.test_kalman_nclt(ks_v, train_loader, plots=False)
        val_loss = (val_loss * dataset_val.total_len() + train_loss * dataset_train.total_len())/(dataset_val.total_len() + dataset_train.total_len())

    return test_loss

def synthetic_hybrid(args, sigma=1, lamb=1, val_on_train=False, load = True,test_time=False):
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    args.device = device
    print("working on device %s" % device)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    dataset_test = synthetic_KNet.SYNTHETIC(partition='test', tr_tt=args.tr_samples, val_tt=args.val_samples, test_tt=args.test_samples,
                                         equations="canonical", gnn_format=True, load = load)
    test_loader = DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False)

    dataset_train = synthetic_KNet.SYNTHETIC(partition='train', tr_tt=args.tr_samples, val_tt=args.val_samples, test_tt=args.test_samples,
                                        equations="canonical", gnn_format=True, load = load)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)

    dataset_val = synthetic_KNet.SYNTHETIC(partition='val', tr_tt=args.tr_samples, val_tt=args.val_samples, test_tt=args.test_samples,
                                      equations="canonical", gnn_format=True, load = load)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)

    # (A, H, Q, R) = synthetic.create_model_parameters_v(s2_x=sigma ** 2, s2_y=sigma ** 2, lambda2=lamb ** 2)
    (A, H, Q, R) = synthetic_KNet.create_model_parameters_canonical(r=lamb,q=sigma)
    
    if test_time:### testing
        print("Start testing")
        try:
            net = torch.load(args.path_results+'best-model.pt', map_location=device)
        except:
            print("Error loading the trained model!!!")
        test_mse = test.test_gnn_kalman(args, net, device, test_loader, plots=False,test_time=True)

        print("Test loss: %.4f" % (test_mse))
        return test_mse.item()
    
    else:### training  
        try:        
            net = torch.load(args.path_results+'best-model.pt', map_location=device)
            print("Load network from previous training")
            NumofParameter = sum(p.numel() for p in net.parameters() if p.requires_grad)
            print("Number of parameters for Hybrid model: ",NumofParameter)  
        except:
            net = gnn.GNN_Kalman(args, A, H, Q, R, dataset_train.x0,  dataset_train.P0, nf=args.nf,
                                prior=args.prior,  learned=args.learned, init=args.init, gamma=args.gamma).to(device)
            print("Initialize Network")
            torch.save(net, args.path_results + 'best-model.pt')          
            NumofParameter = sum(p.numel() for p in net.parameters() if p.requires_grad)
            print("Number of parameters for Hybrid model: ",NumofParameter)    
   
        #print(net)
        optimizer = optim.Adam(net.parameters(), lr=args.lr)

        min_val = test.test_gnn_kalman(args, net, device, val_loader)

        for epoch in range(1, args.epochs + 1):
            #adjust_learning_rate(optimizer, args.lr, epoch)
            train_hybrid(args, net, device, train_loader, optimizer, epoch)

            val_mse = test.test_gnn_kalman(args, net, device, val_loader)
            
            if val_on_train:
                train_mse = test.test_gnn_kalman(args, net, device, train_loader)
                val_mse = (val_mse * dataset_val.total_len() + train_mse * dataset_train.total_len())/(dataset_val.total_len() + dataset_train.total_len())

            if val_mse < min_val:
                min_val = val_mse
                ### save best model on validation set
                torch.save(net, args.path_results + 'best-model.pt')
        return min_val.item()


def main_synhtetic_kalman(args, sigma=0.1, lamb=0.5, val_on_train=False, optimal=False, load = True):

    if val_on_train:
        dataset_train = synthetic_KNet.SYNTHETIC(partition='train', tr_tt=args.tr_samples, val_tt=args.val_samples,
                                            test_tt=args.test_samples,
                                            equations="canonical", gnn_format=True, load = load)
        train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False)

    dataset_val = synthetic_KNet.SYNTHETIC(partition='val', tr_tt=args.tr_samples, val_tt=args.val_samples, test_tt=args.test_samples,
                                      equations="canonical", gnn_format=True, load = load)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)

    dataset_test = synthetic_KNet.SYNTHETIC(partition='test', tr_tt=args.tr_samples, val_tt=args.val_samples, test_tt=args.test_samples,
                                       equations="canonical", gnn_format=True, load = load)
    test_loader = DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False)


    print("Testing for sigma: %.3f \t lambda %.3f" % (sigma, lamb))
    (A, H, Q, R) = synthetic_KNet.create_model_parameters_canonical(q=sigma, r=lamb)
    ks_v = kalman.KalmanSmoother(A, H, Q, R, m1_0, m2_0)
    
    print('Testing Kalman Smoother Canonical')
    val_loss = test.test_kalman_nclt(ks_v, val_loader, plots=False)
    test_loss = test.test_kalman_nclt(ks_v, test_loader, plots=False)

    if val_on_train:
        train_loss = test.test_kalman_nclt(ks_v, train_loader, plots=False)
        val_loss = (val_loss * dataset_val.total_len() + train_loss * dataset_train.total_len())/(dataset_val.total_len() + dataset_train.total_len())

    return val_loss, test_loss


def main_lorenz_hybrid(args, sigma=2, lamb=0.5, val_on_train=False, dt=0.02, K=1, plot_lorenz=False,decimation=True,test_time=False):
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    args.device = device
    print("working on device %s" % device)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    dataset_test = lorenz_KNet.LORENZ(partition='test', max_len=lor_T_test, tr_tt=args.tr_samples, val_tt=args.val_samples, test_tt=args.test_samples, gnn_format=True, sparse=True, sample_dt=dt,decimation=decimation)
    test_loader = DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False)

    dataset_train = lorenz_KNet.LORENZ(partition='train', max_len=lor_T, tr_tt=args.tr_samples, val_tt=args.val_samples, test_tt=args.test_samples, gnn_format=True, sparse=True, sample_dt=dt,decimation=decimation)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)

    dataset_val = lorenz_KNet.LORENZ(partition='val',max_len=lor_T, tr_tt=args.tr_samples, val_tt=args.val_samples, test_tt=args.test_samples, gnn_format=True, sparse=True, sample_dt=dt,decimation=decimation)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)

    if test_time:### testing
        print("Start testing")
        try:
            net = torch.load(args.path_results+'best-model.pt', map_location=device)
        except:
            print("Error loading the trained model!!!")
        test_mse = test.test_gnn_kalman(args, net, device, test_loader, plots=False, plot_lorenz=plot_lorenz,test_time=True)
        
        print("Test loss: %.4f" % (test_mse))
        return test_mse.item()
    
    else:### training      
        # try:        
        #     net = torch.load(args.path_results+'best-model.pt', map_location=device)
        #     print("Load network from previous training")
        #     NumofParameter = sum(p.numel() for p in net.parameters() if p.requires_grad)
        #     print("Number of parameters for Hybrid model: ",NumofParameter)  
        # except:
        net = gnn.Hybrid_lorenz(args, sigma=sigma, lamb=lamb, nf=args.nf, dt=dt, K=K, prior=args.prior, learned=args.learned, init=args.init, gamma=args.gamma).to(device)
        print("Initialize Network")
        torch.save(net, args.path_results + 'best-model.pt')          
        NumofParameter = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print("Number of parameters for Hybrid model: ",NumofParameter)
        
        optimizer = optim.Adam(net.parameters(), lr=args.lr)

        min_val = test.test_gnn_kalman(args, net, device, val_loader)
        # best_test = test.test_gnn_kalman(args, net, device, test_loader, plots=False, plot_lorenz=plot_lorenz)
        for epoch in range(1, args.epochs + 1):
            #adjust_learning_rate(optimizer, args.lr, epoch)
            train_hybrid(args, net, device, train_loader, optimizer, epoch)

            if epoch % args.test_every == 0:
                val_mse = test.test_gnn_kalman(args, net, device, val_loader)
                
                if val_on_train:
                    train_mse = test.test_gnn_kalman(args, net, device, train_loader)
                    val_mse = (val_mse * dataset_val.total_len() + train_mse * dataset_train.total_len())/(dataset_val.total_len() + dataset_train.total_len())

                if val_mse < min_val:
                    min_val = val_mse
                    ### save best model on validation set
                    torch.save(net, args.path_results + 'best-model.pt')
        
        return min_val.item()


def main_lorenz_kalman(args, sigma=2, lamb=0.5, K=1, dt=0.02, val_on_train=False, plots=False, decimation=True):
    if val_on_train:
        dataset_train = lorenz_KNet.LORENZ(partition='train',max_len=lor_T, tr_tt=args.tr_samples, val_tt=args.val_samples,
                                      test_tt=args.test_samples, gnn_format=True, sparse=True, sample_dt=dt,decimation=decimation)
        loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    dataset_test = lorenz_KNet.LORENZ(partition='test', max_len=lor_T_test, tr_tt=args.tr_samples, val_tt=args.val_samples, test_tt=args.test_samples, gnn_format=True, sparse=True, sample_dt=dt,decimation=decimation)
    loader_test = DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False)

    dataset_val = lorenz_KNet.LORENZ(partition='val',max_len=lor_T, tr_tt=args.tr_samples, val_tt=args.val_samples, test_tt=args.test_samples, gnn_format=True, sparse=True, sample_dt=dt,decimation=decimation)
    loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True)

    print("Testing for sigma: %.3f \t lambda %.3f" % (sigma, lamb))
    ks_v = kalman.ExtendedKalman_lorenz(K=K, sigma=sigma, lamb=lamb, dt=dt)
    print('Testing Kalman Smoother A')
    val_loss = test.test_kalman_lorenz(args, ks_v, loader_val, plots=False)
    test_loss = test.test_kalman_lorenz(args, ks_v, loader_test, plots=plots)

    if val_on_train:
        train_loss = test.test_kalman_lorenz(args, ks_v, loader_train, plots=False)
        val_loss = (val_loss * dataset_val.total_len() + train_loss * dataset_train.total_len())/(dataset_val.total_len() + dataset_train.total_len())

    return val_loss, test_loss


def main_nclt_hybrid(args, sx=0.15, sy=0.15, lamb=0.5, val_on_train=False):
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    args.device = device
    print("working on device %s" % device)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    dataset_train = nclt.NCLT(date='2012-01-22', partition='train', ratio=args.nclt_ratio)
    train_loader = DataLoader(dataset_train, batch_size=args.test_batch_size, shuffle=False)

    dataset_val = nclt.NCLT(date='2012-01-22', partition='val', ratio=args.nclt_ratio)
    val_loader = DataLoader(dataset_val, batch_size=args.test_batch_size, shuffle=False)

    dataset_test = nclt.NCLT(date='2012-01-22', partition='test', ratio=1.0)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    (A, H, Q, R) = synthetic_KNet.create_model_parameters_v(s2_x=sx ** 2,  s2_y=sy ** 2, lambda2=lamb ** 2)
    net = gnn.GNN_Kalman(args, A, H, Q, R, settings.x0_v, 0 * np.eye(len(settings.x0_v)), nf=args.nf,
                         prior=args.prior,  learned=args.learned, init=args.init).to(device)

    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    min_val = 1e8
    best_test = 1e8
    for epoch in range(1, args.epochs + 1):
        #adjust_learning_rate(optimizer, args.lr, epoch)
        train_hybrid(args, net, device, train_loader, optimizer, epoch)

        val_mse = test.test_gnn_kalman(args, net, device, val_loader)

        test_mse = test.test_gnn_kalman(args, net, device, test_loader)

        if val_on_train:
            train_mse = test.test_gnn_kalman(args, net, device, train_loader)
            val_mse = (val_mse * dataset_val.total_len() + train_mse * dataset_train.total_len())/(dataset_val.total_len() + dataset_train.total_len())

        if val_mse < min_val:
            min_val, best_test = val_mse, test_mse

    print("Test loss: %.4f" % (best_test))
    return min_val.item(), best_test.item()


def main_nclt_kalman(args, sx=0.15, sy=0.15, lamb=0.5, val_on_train=False):
    if val_on_train:
        dataset_train = nclt.NCLT(date='2012-01-22', partition='train')
        loader_train = DataLoader(dataset_train, batch_size=1, shuffle=False)

    dataset_val = nclt.NCLT(date='2012-01-22', partition='val')
    loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False)

    dataset_test = nclt.NCLT(date='2012-01-22', partition='test')
    loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    print("Testing for sigma: (%.3f, %.3f) \t lambda %.3f" % (sx, sy, lamb))
    (A, H, Q, R) = synthetic_KNet.create_model_parameters_v(T=1., s2_x=sx ** 2, s2_y=sy ** 2, lambda2=lamb ** 2)
    ks_v = kalman.KalmanSmoother(A, H, Q, R, settings.x0_v, 0 * np.eye(len(settings.x0_v)))
    print('Testing Kalman Smoother A')
    val_loss = test.test_kalman_nclt(ks_v, loader_val, plots=False)
    test_loss = test.test_kalman_nclt(ks_v, loader_test, plots=False)

    if val_on_train:
        train_loss = test.test_kalman_nclt(ks_v, loader_train, plots=False)
        val_loss = (val_loss * dataset_val.total_len() + train_loss * dataset_train.total_len())/(dataset_val.total_len() + dataset_train.total_len())

    return val_loss, test_loss


if __name__ == '__main__':
    args = settings.get_settings()

    # main_kalman()
    # main_gnn()
    # main_synthetic_hybrid()
    # main_nclt_test(args)
    # main_nclt_hybrid(args)
    main_lorenz_hybrid(args)


