import numpy as np
import generateModel as gm
import networkx as nx
import pandas as pd
from contextlib import contextmanager
import sys
import os
import logging

sys.path.append('../../')
import modelling

#Suppress logging messages
logging.disable(logging.CRITICAL)
#Supress progressbar
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

threads = 8
repeat_1 = 5
repeat_2 = 100

#Test Erdos-Renyi
print("Testing ER code:")
fault = False
for x in range(repeat_1):
    n = np.random.randint(100,200)
    p = np.random.random()
    edges = [len(gm.ER(n,p,threads)['row']) for i in range(repeat_2)]
    average_p = np.mean(edges)/(n*(n-1))
    if abs(average_p-p) >= 0.01:
        print("Caution: expected p = "+str(p)+" average edge prob = "+str(average_p))
        print("If difference is large there may be an problem with the ER code or installation.")
        fault = True


if not fault:
   print("ER code working correctly")

print()
print('#################################')
print()

#Test SBM
print("Testing SBM code:")
fault = False
for x in range(repeat_1):
    n = np.random.randint(100,200)
    m = np.random.randint(5,15)
    p = np.random.random((m,m))
    actual_p = np.zeros((m,m))
    partition = [np.random.randint(m) for i in range(n)]
    part_size = np.bincount(partition)
    for y in range(repeat_2):
        G = gm.SBM(n, p, partition, threads)
        for i in range(len(G['row'])):
            actual_p[partition[G['row'][i]]][partition[G['col'][i]]] += 1

    actual_p = actual_p/repeat_2
    for a in range(m):
        for b in range(m):
            actual_p[a][b] = actual_p[a][b]/(part_size[a]*part_size[b])
    if np.sum(abs(actual_p-p))/m**2 >= 0.05:
        print("Caution: expected p = "+str(p)+" actual p = "+str(actual_p))
        print("If difference is large there may be an problem with the SBM code or installation.")
        fault = True


if not fault:
   print("SBM code working correctly")

print()
print('#################################')
print()

#Test DD2
print("Testing DD2 code:")
fault = False
for x in range(repeat_1):
    n = np.random.randint(100,200)
    p = np.random.random()
    G_gm = gm.ER(n,p,threads)
    adj_matrix = np.zeros((n,n))
    for i in range(len(G_gm['row'])):
        adj_matrix[G_gm['row'][i]][G_gm['col'][i]] = 1
    xyz = np.random.uniform(low=0,high=200,size=(n,3))
    xyz_df = pd.DataFrame(xyz,columns=['x','y','z'])
    config_dict = {'model_name': 'ConnProb2ndOrder',  # Name of the model (to be used in file names, ...)
                   'model_order': 2,                  # Model order
                   'bin_size_um': 100,                # Bin size (um) for depth binning
                   'max_range_um': None,              # Max. distance (um) range to consider (None to use full distance range)
                   'sample_size': None,               # Size of random subset of neurons to consider (0 or None to disable subsampling)
                   'sample_seed': 4321,               # Seed for selecting random subset of neurons
                   'model_dir': None,                 # Output directory where to save the model (None to disable saving)
                   'data_dir': None,                  # Output directory where to save the extracted data (None to disable saving)
                   'do_plot': False,                  # Enable/disable output plotting
                   'plot_dir': None,                  # Output directory where to save the plots (None to disable saving)
                   'N_split': None}
    with suppress_stdout():
        data_dict, model_dict = modelling.run_model_building(adj_matrix, xyz_df, **config_dict)
    a = model_dict['model_params']['exp_model_scale']
    b = model_dict['model_params']['exp_model_exponent']
    a_vals = []
    b_vals = []
    for y in range(repeat_2):
        G_gm = gm.DD2(n,model_dict['model_params']['exp_model_scale'],model_dict['model_params']['exp_model_exponent'],xyz,threads)
        G_dense = np.zeros((n,n))
        for i in range(len(G_gm['row'])):
            G_dense[G_gm['row'][i]][G_gm['col'][i]] = 1
        with suppress_stdout():
            data_dict_2, model_dict_2 = modelling.run_model_building(G_dense, xyz_df, **config_dict)
        a_vals.append(model_dict_2['model_params']['exp_model_scale'])
        b_vals.append(model_dict_2['model_params']['exp_model_exponent'])
    a_av = np.mean(a_vals)
    b_av = np.mean(b_vals)
    if abs(a-a_av) >= 0.05  or abs(b-b_av) >= 0.005:
        print("Caution: expected a,b = "+str(a)+","+str(b)+" actual a,b = "+str(a_av)+","+str(b_av))
        print("If difference is large there may be an problem with the DD2 code or installation.")
        fault = True


if not fault:
   print("DD2 code working correctly")

print()
print('#################################')
print()

#Test DD3
print("Testing DD3 code:")
fault = False
a_diff = 0.1
b_diff = 0.005
for x in range(repeat_1):
    n = np.random.randint(100,200)
    p = np.random.random()
    G_gm = gm.ER(n,p,threads)
    adj_matrix = np.zeros((n,n))
    for i in range(len(G_gm['row'])):
        adj_matrix[G_gm['row'][i]][G_gm['col'][i]] = 1
    xyz = np.random.uniform(low=0,high=200,size=(n,3))
    depth = np.random.uniform(low=0,high=200,size=(n,))
    xyz_df = pd.DataFrame(xyz,columns=['x','y','z'])
    xyz_df['depth'] = depth
    config_dict = {'model_name': 'ConnProb3rdOrder',  # Name of the model (to be used in file names, ...)
                   'model_order': 3,                  # Model order
                   'bin_size_um': 100,                # Bin size (um) for depth binning
                   'max_range_um': None,              # Max. distance (um) range to consider (None to use full distance range)
                   'sample_size': None,               # Size of random subset of neurons to consider (0 or None to disable subsampling)
                   'sample_seed': 4321,               # Seed for selecting random subset of neurons
                   'model_dir': None,                 # Output directory where to save the model (None to disable saving)
                   'data_dir': None,                  # Output directory where to save the extracted data (None to disable saving)
                   'do_plot': False,                  # Enable/disable output plotting
                   'plot_dir': None,                  # Output directory where to save the plots (None to disable saving)
                   'N_split': None}
    with suppress_stdout():
        data_dict, model_dict = modelling.run_model_building(adj_matrix, xyz_df, **config_dict)
    params = (model_dict['model_params']['bip_neg_exp_model_scale'],model_dict['model_params']['bip_neg_exp_model_scale'],model_dict['model_params']['bip_pos_exp_model_scale'],model_dict['model_params']['bip_pos_exp_model_exponent'])
    param_vals = []
    for y in range(repeat_2):
        G_gm = gm.DD3(n,params[0],params[1],params[2],params[3],xyz,depth,threads)
        G_dense = np.zeros((n,n))
        for i in range(len(G_gm['row'])):
            G_dense[G_gm['row'][i]][G_gm['col'][i]] = 1
        with suppress_stdout():
            data_dict_2, model_dict_2 = modelling.run_model_building(G_dense, xyz_df, **config_dict)
        param_vals.append((model_dict_2['model_params']['bip_neg_exp_model_scale'],model_dict_2['model_params']['bip_neg_exp_model_scale'],model_dict_2['model_params']['bip_pos_exp_model_scale'],model_dict_2['model_params']['bip_pos_exp_model_exponent']))
    params_av = np.mean(param_vals,axis=0)
    if abs(params[0]-params_av[0]) >= a_diff or abs(params[2]-params_av[2]) >= a_diff or abs(params[1]-params_av[1]) >= b_diff or abs(params[3]-params_av[3]) >= b_diff:
        print("Caution: expected aN,bN,aP,bP = "+str(params)+" actual values = "+str(params_av))
        print("If difference is large there may be an problem with the DD3 code or installation.")
        fault = True


if not fault:
   print("DD3 code working correctly")
