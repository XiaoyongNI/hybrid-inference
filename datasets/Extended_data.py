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
wandb_switch = True #True for wandb, False for no wandb
InitIsRandom = True
RotateH = False
RotateF = False
HNL = False #True for Non-linear observation h, False for linear H
CV_model = False #True for CV model, False for CA model

# compact_path_linear = "simulations/Linear/Scaling_to_large_models/5x5_rq020_T20.pt" # path to load pre-generated dataset
compact_path_linear = 'simulations/Linear/Linear_CA/New_decimated_dt1e-2_T100_r0_randnInit.pt'
decimation = True # true for decimation case, false for DT case
compact_path_lor_decimation = "simulations/LA/decimation/decimated_r0_Ttest3000.pt"
compact_path_lor_DT = "simulations/LA/DT/T100_Hrot1/data_lor_v20_rq-1010_T100.pt"
r2 = 1
r = np.sqrt(r2) # lamb
vdB = -20 # ratio v=q2/r2
v = 10**(vdB/10)
q2 = np.multiply(v,r2)
q = np.sqrt(q2) # sigma

if decimation:
    # opt_1overq2_dB = 8.2391
    # opt_q = np.sqrt(10**(-opt_1overq2_dB/10))
    opt_q = np.sqrt(0.1)
else:
    opt_q = q #DT case

########################
### DataSet Settings ###
########################

# Number of Training Examples
N_E = 1000

# Number of Cross Validation Examples
N_CV = 100

# Number of Test Examples
N_T = 200

# Sequence Length for Linear case
T = 100
T_test = 100
# Sequence Length for NL lorenz case
lor_T = 100
lor_T_test = 100

# Init condition and delta_t for Lorenz
m1_x0 = np.array([1., 1., 1.]).astype(np.float32) # initial x0
m2_x0 = np.diag([1] * 3) * 0 # initial P0
delta_t = 0.02 # dt that generates the dataset
sample_delta_t = 0.02 # sampling dt

lr_coeff = 1 # the ratio between GM message and GNN message

###########################
### Non-linear case h/H ###
###########################

def h_nonlinear(x):
    return torch.squeeze(toSpherical(x))

def toSpherical(cart):

    rho = torch.norm(cart, p=2).view(1,1)
    phi = torch.atan2(cart[1, ...], cart[0, ...]).view(1, 1)
    phi = phi + (phi < 0).type_as(phi) * (2 * torch.pi)

    theta = torch.acos(cart[2, ...] / rho).view(1, 1)

    spher = torch.cat([rho, theta, phi], dim=0)

    return spher

H_design = np.diag([1]*3)
## Angle of rotation in the 3 axes
roll_deg = yaw_deg = pitch_deg = 1

roll = roll_deg * (math.pi/180)
yaw = yaw_deg * (math.pi/180)
pitch = pitch_deg * (math.pi/180)

RX = np.array([
                [1, 0, 0],
                [0, math.cos(roll), -math.sin(roll)],
                [0, math.sin(roll), math.cos(roll)]])
RY = np.array([
                [math.cos(pitch), 0, math.sin(pitch)],
                [0, 1, 0],
                [-math.sin(pitch), 0, math.cos(pitch)]])
RZ = np.array([
                [math.cos(yaw), -math.sin(yaw), 0],
                [math.sin(yaw), math.cos(yaw), 0],
                [0, 0, 1]])

RotMatrix = np.matmul(np.matmul(RZ, RY), RX)
H_lor_rotated = np.matmul(RotMatrix,H_design)



####################################
## Linear Canonical Case F and H ###
####################################
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

# Inaccurate model knowledge based on matrix rotation
F_rotated = np.zeros_like(F)
H_rotated = np.zeros_like(H)
if(m==2):
    alpha_degree = 10
    rotate_alpha = np.array([alpha_degree/180*np.pi])
    cos_alpha = np.cos(rotate_alpha)
    sin_alpha = np.sin(rotate_alpha)
    rotate_matrix = np.squeeze(np.array([[cos_alpha, -sin_alpha],
                                [sin_alpha, cos_alpha]]))
    # print(rotate_matrix)
    F_rotated = np.matmul(F,rotate_matrix) #inaccurate process model
    H_rotated = np.matmul(H,rotate_matrix) #inaccurate observation model

if RotateH:
    H = np.squeeze(H_rotated)

if RotateF:
    F = np.squeeze(F_rotated)


################################
### 5 x 5, 10 x 10 and so on ###
################################
# m = 5
# n = 5
# ## F in canonical form
# F = np.eye(m)
# F[0] = np.ones((1,m))

# ## H in reverse canonical form
# H = np.zeros((n,n))
# H[0] = np.ones((1,n))
# for i in range(n):
#     H[i,n-1-i] = 1

# m1_0 = np.zeros((m), dtype=np.float32)
# m2_0 = 0 * 0 * np.eye(m)

#############################
## Linear CA(and CV) Case ###
#############################
CA_m = 3 # dim of state
CV_m = 2 # dim of state for CV model
CA_n = 1 # dim of observation
std = 1
CA_m1_0 = np.zeros((CA_m), dtype=np.float32) # Initial State
CV_m1_0 = np.zeros((CV_m), dtype=np.float32) # Initial State for CV
CA_m2_0 = std * std * np.eye(CA_m) # Initial Covariance
CV_m2_0 = std * std * np.eye(CV_m) # Initial Covariance for CV

delta_t_gen =  1e-2

### state evolution matrix F and observation matrix H
F_gen = np.array([[1, delta_t_gen,0.5*delta_t_gen**2],
                  [0,       1,       delta_t_gen],
                  [0,       0,         1]], dtype=np.float32)

F_CV = np.array([[1, delta_t_gen],
                     [0,           1]], dtype=np.float32)       


# Observe only the postion
H_onlyPos = np.array([[1, 0, 0]], dtype=np.float32)

### process noise Q and observation noise R 
# Noise Parameters
CA_r2 = 1
CA_q2 = 1
CV_q2 = 0.1 # can be tuned

Q_gen = CA_q2 * np.array([[1/20*delta_t_gen**5, 1/8*delta_t_gen**4,1/6*delta_t_gen**3],
                           [ 1/8*delta_t_gen**4, 1/3*delta_t_gen**3,1/2*delta_t_gen**2],
                           [ 1/6*delta_t_gen**3, 1/2*delta_t_gen**2,       delta_t_gen]], dtype=np.float32)

Q_CV = CA_q2 * np.array([[1/3*delta_t_gen**3, 1/2*delta_t_gen**2],
                          [1/2*delta_t_gen**2, delta_t_gen]], dtype=np.float32) 

R_onlyPos = CA_r2 




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
