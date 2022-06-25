import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import torch.utils.data as data
import torch
import os
import pickle
from mpl_toolkits.mplot3d import Axes3D
import math
from datasets.Extended_data import compact_path_lor_decimation, compact_path_lor_DT, lor_T, Decimate_and_perturbate_Data, delta_t, sample_delta_t, N_E, N_CV, N_T


class LORENZ(data.Dataset):
    def __init__(self, partition='train', max_len=lor_T, tr_tt=1000, val_tt=1000, test_tt=3000, gnn_format=False, sparse=True, sample_dt=0.02, decimation=True):
        self.partition = partition  # training set or test set
        self.max_len = max_len
        self.gnn_format = gnn_format
        self.sparse = sparse
        self.lamb = 1
        self.x0 = [1.0, 1.0, 1.0]
        self.H = np.diag([1]*3)
        self.R = np.diag([1]*3) * self.lamb ** 2
        self.sample_dt = sample_delta_t
        self.dt = delta_t

        self.rho = 28.0
        self.sigma = 10.0
        self.beta = 8.0 / 3.0

        if decimation:
            print("load decimated Lorenz Attractor data!")
            data_gen_file = torch.load(compact_path_lor_decimation,map_location=torch.device("cpu"))
            [true_sequence] = data_gen_file['All Data']
            ### self.data = list[state, meas]
            if self.partition == 'train':
                [train_target, train_input] = Decimate_and_perturbate_Data(true_sequence, self.dt, self.sample_dt, N_E, self.h, self.lamb)
                ### Convert to np array and float64 type
                train_input = train_input.numpy().astype(np.float64)
                train_target = train_target.numpy().astype(np.float64)
                ### Convert to list 
                datalist = []
                if tr_tt < max_len:
                    n_e = random.randint(0, N_E - 1)
                    datalist.append([np.transpose(train_target[n_e,:,0:tr_tt],(1,0)), np.transpose(train_input[n_e,:,0:tr_tt],(1,0))])
                else:
                    for i in range(np.minimum(train_target.shape[0],int(tr_tt/max_len))):                   
                        datalist.append([np.transpose(train_target[i,:,:],(1,0)), np.transpose(train_input[i,:,:],(1,0))])
                self.data = datalist

            if self.partition == 'val':
                [cv_target, cv_input] = Decimate_and_perturbate_Data(true_sequence, self.dt, self.sample_dt, N_CV, self.h, self.lamb)
                ### Convert to np array and float64 type
                cv_input = cv_input.numpy().astype(np.float64)
                cv_target = cv_target.numpy().astype(np.float64)
                ### Convert to list 
                datalist = []
                for i in range(cv_target.shape[0]):                   
                    datalist.append([np.transpose(cv_target[i,:,:],(1,0)), np.transpose(cv_input[i,:,:],(1,0))])
                self.data = datalist

            if self.partition == 'test':
                [test_target, test_input] = Decimate_and_perturbate_Data(true_sequence, self.dt, self.sample_dt, N_T, self.h, self.lamb)
                ### Convert to np array and float64 type
                test_input = test_input.numpy().astype(np.float64)
                test_target = test_target.numpy().astype(np.float64)
                ### Convert to list 
                datalist = []
                for i in range(test_target.shape[0]):                   
                    datalist.append([np.transpose(test_target[i,:,:],(1,0)), np.transpose(test_input[i,:,:],(1,0))])
                self.data = datalist
                
        else:           
            print("load DT Lorenz Attractor data:"+compact_path_lor_DT)
            [train_input,train_target, cv_input, cv_target, test_input, test_target] =  torch.load(compact_path_lor_DT,map_location=torch.device("cpu"))
            ### self.data = list[state, meas]
            if self.partition == 'train':
                ### Convert to np array and float64 type
                train_input = train_input.numpy().astype(np.float64)
                train_target = train_target.numpy().astype(np.float64)
                ### Convert to list
                datalist = []
                if tr_tt < max_len:
                    n_e = random.randint(0, N_E - 1)
                    datalist.append([np.transpose(train_target[n_e,:,0:tr_tt],(1,0)), np.transpose(train_input[n_e,:,0:tr_tt],(1,0))])
                else:
                    for i in range(np.minimum(train_target.shape[0],int(tr_tt/max_len))):                   
                        datalist.append([np.transpose(train_target[i,:,:],(1,0)), np.transpose(train_input[i,:,:],(1,0))])
                self.data = datalist

            if self.partition == 'val':
                ### Convert to np array and float64 type
                cv_input = cv_input.numpy().astype(np.float64)
                cv_target = cv_target.numpy().astype(np.float64)
                ### Convert to list 
                datalist = []
                for i in range(cv_target.shape[0]):                   
                    datalist.append([np.transpose(cv_target[i,:,:],(1,0)), np.transpose(cv_input[i,:,:],(1,0))])
                self.data = datalist

            if self.partition == 'test':
                ### Convert to np array and float64 type
                test_input = test_input.numpy().astype(np.float64)
                test_target = test_target.numpy().astype(np.float64)
                ### Convert to list 
                datalist = []
                for i in range(test_target.shape[0]):                   
                    datalist.append([np.transpose(test_target[i,:,:],(1,0)), np.transpose(test_input[i,:,:],(1,0))])
                self.data = datalist

        self._generate_operators()

        '''
        tr_samples = int(tr_tt/max_len)
        test_samples = int(test_tt / max_len)
        val_samples = int(val_tt / max_len)
        if self.partition == 'train':
            self.data = [self._generate_sample(i, max_len) for i in range(test_samples, test_samples + tr_samples)]
        elif self.partition == 'val':
            self.data = [self._generate_sample(i, max_len) for i in range(test_samples + tr_samples, test_samples + tr_samples + val_samples)]
        elif self.partition == 'test':
            self.data = [self._generate_sample(i, max_len) for i in range(test_samples)]
        else:
            raise Exception('Wrong partition')
        '''

        print("%s partition created, \t num_samples %d \t num_timesteps: %d" % (
        self.partition, len(self.data), self.total_len()))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (state, meas) where target is index of the target class.
        """
        state, meas, operator = self.data[index]
        x0 = state[0]
        P0 = np.eye(x0.shape[0])
        if self.gnn_format:
            return np.arange(0, meas.shape[0], 1), state.astype(np.float32), meas.astype(np.float32), x0, P0, operator
        else:
            return state, meas,  self.x0, self.P0

    def __len__(self):
        return len(self.data)

    def dump(self, path, object):
        if not os.path.exists('temp'):
            os.makedirs('temp')
        with open(path, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)


    def _split_data(self):
        num_splits = math.ceil(float(self.data[0].shape[0])/self.max_len)
        data = []
        for i in range(int(num_splits)):
            i_start = i*self.max_len
            i_end = (i+1)*self.max_len
            data.append([self.data[0][i_start:i_end], self.data[1][i_start:i_end]])
        self.data = data

    def _generate_operators(self):
        for i in range(len(self.data)):
            tt = self.data[i][0].shape[0]
            self.data[i].append(self.__buildoperators_sparse(tt))

    def _generate_sample(self, seed, tt):
        np.random.seed(seed)
        sample = self._simulate_system(tt=tt, x0=self.x0)

        # returns state, measurement
        return list(sample)

    def f(self, state, t):
        x, y, z = state  # unpack the state vector
        return self.sigma * (y - x), x * (self.rho - z) - y, x * y - self.beta * z  # derivatives
    
    def h(self, x):
        H_tensor = torch.from_numpy(np.asarray(self.H, dtype=np.float32))
        y = torch.matmul(H_tensor, x)
        return y

    def _simulate_system(self, tt, x0):
        t = np.arange(0.0, tt*self.sample_dt, self.dt)
        states = odeint(self.f, x0, t)
        states_ds = np.zeros((tt, 3))
        for i in range(states_ds.shape[0]):
            states_ds[i] = states[i*int(self.sample_dt/self.dt)]
        states = states_ds

        #Measurement
        meas_model = MeasurementModel(self.H, self.R)
        meas = np.zeros(states.shape)
        for i in range(len(states)):
            meas[i] = meas_model(states[i])
        return states, meas


    def __build_operators(self, nn=20):
        # Identity
        I = np.expand_dims(np.eye(nn), 2)

        #Messages
        mr = np.pad(I, ((1, 0), (0, 0), (0, 0)), 'constant', constant_values=(0))[0:nn, :, :]
        ml = np.pad(I, ((0, 1), (0, 0), (0, 0)), 'constant', constant_values=(0))[1:(nn+1), :, :]

        return np.concatenate([I, mr, ml], axis=2).astype(np.float32)

    def __buildoperators_sparse_old(self, nn=20):
        # Identity
        i = torch.LongTensor([[i,i] for i in range(nn)])
        v = torch.FloatTensor([1 for i in range(nn)])
        I = torch.sparse.FloatTensor(i.t(), v)

        #Message right
        i = torch.LongTensor([[i, i+1] for i in range(nn-1)] + [[nn-1, nn-1]])
        v = torch.FloatTensor([1 for i in range(nn-1)] + [0])
        mr = torch.sparse.FloatTensor(i.t(), v)

        #Message left
        i = torch.LongTensor([[0, nn-1]] + [[i+1, i] for i in range(nn-1)])
        v = torch.FloatTensor([0] + [1 for i in range(nn-1)])
        ml = torch.sparse.FloatTensor(i.t(), v)

        return [I, mr, ml]

    def __buildoperators_sparse(self, nn=20):
        # Message right to left
        m_left_r = []
        m_left_c = []

        m_right_r = []
        m_right_c = []

        m_up_r = []
        m_up_c = []

        for i in range(nn - 1):
            m_left_r.append(i)
            m_left_c.append((i + 1))

            m_right_r.append(i + 1)
            m_right_c.append((i))

        for i in range(nn):
            m_up_r.append(i)
            m_up_c.append(i + nn)

        m_left = [torch.LongTensor(m_left_r), torch.LongTensor(m_left_c)]
        m_right = [torch.LongTensor(m_right_r), torch.LongTensor(m_right_c)]
        m_up = [torch.LongTensor(m_up_r), torch.LongTensor(m_up_c)]

        return {"m_left": m_left, "m_right": m_right, "m_up": m_up}


    def total_len(self):
        total = 0
        for state, meas, _ in self.data:
            total += meas.shape[0]
        return total


def __plot_trajectory(states):
    fig = plt.figure(linewidth=0.0)
    ax = fig.gca(projection='3d')
    ax.plot(states[:, 0], states[:, 1], states[:, 2], linewidth=0.5)
    plt.axis('off')
    plt.show()

def plot_trajectory(args, states, mse=0.):
    if not os.path.exists('plots'):
        os.makedirs('plots')

    if args.learned and args.prior:
        path = 'hybrid'
    elif args.learned and not args.prior:
        path = 'learned'
    elif not args.learned and args.prior:
        path = 'prior'
    else:
        path = 'baseline'

    if not os.path.exists('plots/%s' % path):
        os.makedirs('plots/%s' % path)

    fig = plt.figure(linewidth=0.0)
    ax = fig.gca(projection='3d')
    ax.plot(states[:, 0], states[:, 1], states[:, 2], linewidth=0.5)
    plt.axis('off')
    plt.savefig('plots/%s/tr_samples_%d_loss_%.4f.png' % (path, args.tr_samples, mse))

class MeasurementModel():
    def __init__(self, H, R):
        self.H = H
        self.R = R

        (n, _) = R.shape
        self.zero_mean = np.zeros(n)

    def __call__(self, x):
        measurement = self.H @ x + np.random.multivariate_normal(self.zero_mean, self.R)
        return measurement

if __name__ == '__main__':
    dataset = LORENZ(partition='test', sample_dt=0.01, no_pickle=False, max_len=3000, test_tt=3000, val_tt=0, tr_tt=0)
    __plot_trajectory(dataset.data[0][0])














