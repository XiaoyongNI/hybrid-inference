import torch
import numpy as np
import math
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if torch.cuda.is_available():
   dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
   torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
   dev = torch.device("cpu")
   print("Running on the CPU")

#########################################
### Dataset path and noise statistics ###
#########################################
compact_path_linear = "temp/2x2_rq020_T100_Ttest1000.pt" # path to load pre-generated dataset
compact_path_lor_decimation = "temp/data_gen.pt"
compact_path_lor_DT = "temp/data_lor_v20_rq020_T200.pt"
r2 = 1 
r = np.sqrt(r2) # lamb
vdB = -20 # ratio v=q2/r2
v = 10**(vdB/10)
q2 = np.multiply(v,r2)
q = np.sqrt(q2) # sigma

opt_1overq2_dB = 8.2391
opt_q = np.sqrt(10**(-opt_1overq2_dB/10))

#######################
### Size of DataSet ###
#######################

# Number of Training Examples
N_E = 1000

# Number of Cross Validation Examples
N_CV = 100

# Number of Test Examples
N_T = 200

# Sequence Length
T = 100
T_test = 1000

lor_T = 200
lor_T_test = 200

#################
## Design #10 ###
#################
F10 = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

H10 = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

############
## 2 x 2 ###
############
m = 2
n = 2
F = F10[0:m, 0:m]
H = np.eye(2)
m1_0 = np.array([0.0, 0.0], dtype=np.float32)
# m1x_0_design = torch.tensor([[10.0], [-10.0]])
m2_0 = 0 * 0 * np.eye(m)


#############
### 5 x 5 ###
#############
# m = 5
# n = 5
# F = F10[0:m, 0:m]
# H = H10[0:n, 10-m:10]
# m1_0 = torch.zeros(m, 1).to(cuda0)
# # m1x_0_design = torch.tensor([[1.0], [-1.0], [2.0], [-2.0], [0.0]]).to(cuda0)
# m2_0 = 0 * 0 * torch.eye(m).to(cuda0)

##############
## 10 x 10 ###
##############
# m = 10
# n = 10
# F = F10[0:m, 0:m]
# H = H10
# m1_0 = torch.zeros(m, 1).to(cuda0)
# # m1x_0_design = torch.tensor([[10.0], [-10.0]])
# m2_0 = 0 * 0 * torch.eye(m).to(cuda0)

def DataGen_True(SysModel_data, fileName, T):

    SysModel_data.GenerateBatch(1, T, randomInit=False)
    test_input = SysModel_data.Input
    test_target = SysModel_data.Target

    # torch.save({"True Traj":[test_target],
    #             "Obs":[test_input]},fileName)
    torch.save([test_input, test_target], fileName)

def DataGen(SysModel_data, fileName, T, T_test,randomInit=False):

    ##################################
    ### Generate Training Sequence ###
    ##################################
    SysModel_data.GenerateBatch(N_E, T, randomInit=randomInit)
    training_input = SysModel_data.Input
    training_target = SysModel_data.Target

    ####################################
    ### Generate Validation Sequence ###
    ####################################
    SysModel_data.GenerateBatch(N_CV, T, randomInit=randomInit)
    cv_input = SysModel_data.Input
    cv_target = SysModel_data.Target

    ##############################
    ### Generate Test Sequence ###
    ##############################
    SysModel_data.GenerateBatch(N_T, T_test, randomInit=randomInit)
    test_input = SysModel_data.Input
    test_target = SysModel_data.Target

    #################
    ### Save Data ###
    #################
    torch.save([training_input, training_target, cv_input, cv_target, test_input, test_target], fileName)

def DataLoader(fileName):

    [training_input, training_target, cv_input, cv_target, test_input, test_target] = torch.load(fileName)
    return [training_input, training_target, cv_input, cv_target, test_input, test_target]

def DataLoader_GPU(fileName):
    [training_input, training_target, cv_input, cv_target, test_input, test_target] = torch.utils.data.DataLoader(torch.load(fileName),pin_memory = False)
    training_input = training_input.squeeze().to(dev)
    training_target = training_target.squeeze().to(dev)
    cv_input = cv_input.squeeze().to(dev)
    cv_target =cv_target.squeeze().to(dev)
    test_input = test_input.squeeze().to(dev)
    test_target = test_target.squeeze().to(dev)
    return [training_input, training_target, cv_input, cv_target, test_input, test_target]

def DecimateData(all_tensors, t_gen,t_mod, offset=0):
    
    # ratio: defines the relation between the sampling time of the true process and of the model (has to be an integer)
    ratio = round(t_mod/t_gen)

    i = 0
    all_tensors_out = all_tensors
    for tensor in all_tensors:
        tensor = tensor[:,(0+offset)::ratio]
        if(i==0):
            all_tensors_out = torch.cat([tensor], dim=0).view(1,all_tensors.size()[1],-1)
        else:
            all_tensors_out = torch.cat([all_tensors_out,tensor], dim=0)
        i += 1

    return all_tensors_out

def Decimate_and_perturbate_Data(true_process, delta_t, delta_t_mod, N_examples, h, lambda_r, offset=0):
    
    # Decimate high resolution process
    decimated_process = DecimateData(true_process, delta_t, delta_t_mod, offset)

    noise_free_obs = getObs(decimated_process,h)

    # Replicate for computation purposes
    decimated_process = torch.cat(int(N_examples)*[decimated_process])
    noise_free_obs = torch.cat(int(N_examples)*[noise_free_obs])


    # Observations; additive Gaussian Noise
    observations = noise_free_obs + torch.randn_like(decimated_process) * lambda_r

    return [decimated_process, observations]

def getObs(sequences, h):
    i = 0
    sequences_out = torch.zeros_like(sequences)
    for sequence in sequences:
        for t in range(sequence.size()[1]):
            sequences_out[i,:,t] = h(sequence[:,t])
    i = i+1

    return sequences_out

def Short_Traj_Split(data_target, data_input, T):
    data_target = list(torch.split(data_target,T,2))
    data_input = list(torch.split(data_input,T,2))
    data_target.pop()
    data_input.pop()
    data_target = torch.squeeze(torch.cat(list(data_target), dim=0))
    data_input = torch.squeeze(torch.cat(list(data_input), dim=0))
    return [data_target, data_input]
