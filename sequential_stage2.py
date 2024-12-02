# -*- coding: utf-8 -*-
"""
Created on Wed July 23 14:01:12 2024

@author: luuk barbian
"""

import random
import os
import time
import sequential_stage1
import setsparameters
from gurobipy import Model,GRB,LinExpr,quicksum
import pickle


start_time = time.time()
random.seed(setsparameters.r_seed)
cwd = os.getcwd()

N_dict = setsparameters.N_dict
N_dict_idx_at = setsparameters.N_dict_idx_at
N_dict_at_idx = setsparameters.N_dict_at_idx

K = setsparameters.K
N_kv = setsparameters.N_kv
N_k = setsparameters.N_k
dict_IATA_idx = setsparameters.dict_IATA_idx
airports_IATA = setsparameters.airports_IATA

G_dict = setsparameters.G_dict
F_dict = setsparameters.F_dict
Source_TSN_dict = setsparameters.Source_TSN_dict
TSN_Sink_dict = setsparameters.TSN_Sink_dict
Source_Sink_dict = setsparameters.Source_Sink_dict
ULD_source_TSN_dict = setsparameters.ULD_source_TSN_dict
TSN_ULD_sink_dict = setsparameters.TSN_ULD_sink_dict
A_all_dict_3 = setsparameters.A_all_dict_3
K_f = setsparameters.K_f
G_u = setsparameters.G_u
F_u = setsparameters.F_u
U_v_f = setsparameters.U_v_f
AK_k_n_plus = setsparameters.AK_k_n_plus
AK_k_n_minus = setsparameters.AK_k_n_minus
N_K_realTSN = setsparameters.N_K_realTSN
AU_u_n_plus = setsparameters.AU_u_n_plus
AU_u_n_minus = setsparameters.AU_u_n_minus
F_info = setsparameters.F_info
W_fk = setsparameters.W_fk
earliest_node_airport = setsparameters.earliest_node_airport
latest_node_airport = setsparameters.latest_node_airport
F_r_plus = setsparameters.F_r_plus
F_r_minus = setsparameters.F_r_minus


U_dict = setsparameters.U_dict
U_r = setsparameters.U_r
R_u = setsparameters.R_u
U_uld_type_idx = setsparameters.U_uld_type_idx
V_info = setsparameters.V_info
V_ULD_type_idx_dict = setsparameters.V_ULD_type_idx_dict
V_ULD_type = setsparameters.V_ULD_type

R_dict = setsparameters.R_dict
INC_dict = setsparameters.INC_dict

#dummy_U_dict = setsparameters.dummy_U_dict
#dummy_U_r = setsparameters.dummy_U_r
#dummy_U_f = setsparameters.dummy_U_f
#dummy_AU_u_n_plus = setsparameters.dummy_AU_u_n_plus
#dummy_AU_u_n_minus = setsparameters.dummy_AU_u_n_minus
#dummy_ULD_source_TSN_dict = setsparameters.dummy_ULD_source_TSN_dict
#dummy_TSN_ULD_sink_dict = setsparameters.dummy_TSN_ULD_sink_dict
#dummy_G_u = setsparameters.dummy_G_u
#dummy_F_u = setsparameters.dummy_F_u
#dummy_F_r_plus = setsparameters.dummy_F_r_plus
#dummy_F_r_minus = setsparameters.dummy_F_r_minus
V_info_simp = setsparameters.V_info_simp


schedule_small = setsparameters.schedule_small
clustering_requests = setsparameters.clustering_requests
clustering_maximum = setsparameters.clustering_maximum
subset = setsparameters.subset
max_num_of_ulds_per_R = setsparameters.max_num_of_ulds_per_R
comp_level = setsparameters.comp_level
Payload = setsparameters.Payload
free_aircraft_placement = setsparameters.free_aircraft_placement


X_tuple_first_stage_dict = sequential_stage1.X_tuple_first_stage_dict
F_r_first_stage = sequential_stage1.F_r_first_stage_dict
#%%
if schedule_small:
    folder_name = 'reduced'
if not schedule_small:
    folder_name = 'full'
folder_name = folder_name + '_' + comp_level
folder_name = folder_name + '_'+ str(Payload)

if subset:
    folder_name = folder_name + '_subset' + str(max_num_of_ulds_per_R)
if clustering_requests:
    if clustering_maximum == 0:
        folder_name = folder_name + '_clusteringSmall'
    if clustering_maximum == 1:
        folder_name = folder_name + '_clusteringLarge'
if free_aircraft_placement:
    folder_name = folder_name + '_free'
if not free_aircraft_placement:
    folder_name = folder_name + '_restricted'

folder_name = folder_name + '_seed' + str(setsparameters.r_seed)

print('FOLDER NAME:', folder_name)

#%%
########################################################################################################################
############################################ DEFINING THE MODEL SECOND STAGE ###########################################
########################################################################################################################
# Setup model
stage2 = Model()

# Decision variables

x = {}  # aircraft fleet routing decision variables
y = {}  # ULD routing decision variables
z = {}  # Request to ULD assignment decision variables

# Aircraft routing decision variables for ground arcs (integer)
for g in G_dict.keys():
    for k, v in K.items():
        x[g, k] = stage2.addVar(lb=0, ub=v['#'], vtype=GRB.INTEGER, name='x[%s,%s]' % (g, k))

# Aircraft routing decision variables for flight arcs (binary)
for f in F_dict.keys():
    for k in K_f[f]:
        x[f, k] = stage2.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='x[%s,%s]' % (f, k))

# Aircraft routing decision variables from dummy source to TSN (integer)
for a in Source_TSN_dict.keys():
    for k, v in K.items():
        x[a, k] = stage2.addVar(lb=0, ub=v['#'], vtype=GRB.INTEGER, name='x[%s,%s]' % (a, k))

# Aircraft routing decision variables from TSN to dummy sink (integer)
for a in TSN_Sink_dict.keys():
    for k, v in K.items():
        x[a, k] = stage2.addVar(lb=0, ub=v['#'], vtype=GRB.INTEGER, name='x[%s,%s]' % (a, k))

# Aircraft routing decision variable from dummy source to dummy sink (integer)
for k, v in K.items():
    for a in Source_Sink_dict.keys():
        x[a, k] = stage2.addVar(lb=0, ub=v['#'], vtype=GRB.INTEGER, name='x[%s,%s]' % (a, k))

# dummy ULD routing decision variables from source node to TSN node
for idx, a in enumerate(ULD_source_TSN_dict.keys()):
    y[a, idx] = stage2.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='y[%s,%s]' % (a, idx))

# dummy ULD routing decision variables from TSN node to sink node
for idx, a in enumerate(TSN_ULD_sink_dict.keys()):
    y[a, idx] = stage2.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='y[%s,%s]' % (a, idx))

# dummy ULD routing decision variables for ground arcs
for u in U_dict.keys():
    for g in G_u[u]:
        y[g, u] = stage2.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='y[%s,%s]' % (g, u))

# dummy ULD routing decision variables for flight arcs
for u in U_dict.keys():
    for f in F_u[u]:
        y[f, u] = stage2.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='y[%s,%s]' % (f, u))

# Request to dummy ULD assignment decision variables
for r in R_dict.keys():
    for u in U_r[r]:
        z[r, u] = stage2.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='z[%s,%s]' % (r, u))


#%%
########################################################################################################################
############################################### CONSTRAINTS ############################################################
########################################################################################################################

# For each flight leg, we can carry as many ULDs of type v as there are allowed by the configuration
C2 = stage2.addConstrs((quicksum(y[f, u] for u in U_v_f[(ULD_type_idx, f)]) -
                       quicksum(N_kv[k][ULD_type_idx] * x[f, k] for k in K_f[f]) <= 0
                       for f in F_dict.keys() for ULD_type_idx in U_uld_type_idx.keys()),
                      name="C2")

# For each ULD, we must satisfy the weight capacity
C3 = stage2.addConstrs((quicksum(R_dict[r][5] * z[r, u] for r in R_u[u]) <=
                       V_info[U_dict[u][2]][0] for u in U_dict.keys()),
                      name="C3")

# For each ULD, we must satisfy the volume capacity
C4 = stage2.addConstrs((quicksum(R_dict[r][6] * z[r, u] for r in R_u[u]) <=
                       V_info[U_dict[u][2]][1] for u in U_dict.keys()),
                      name="C4")

# Each request can be assigned to at most one ULD
C5 = stage2.addConstrs((quicksum(z[r, u] for u in U_r[r]) <=
                       1 for r in R_dict.keys()),
                      name="C5")

# Conservation of flow in the TSN per ULD
C9 = stage2.addConstrs((quicksum(y[a, u] for a in AU_u_n_plus[(u, n)]) -
                       quicksum(y[a, u] for a in AU_u_n_minus[(u, n)]) == 0
                       for u in U_dict.keys() for n in N_K_realTSN.keys()), name="C9")

C14 = stage2.addConstrs((z[r, u] -
                        quicksum(y[f, u] for f in F_r_plus[r, u]) <= 0
                        for r in R_dict.keys() for u in U_r[r]), name="C14")

C15 = stage2.addConstrs((z[r, u] -
                        quicksum(y[f, u] for f in F_r_minus[r, u]) <= 0
                        for r in R_dict.keys() for u in U_r[r]), name="C15")

# Incompatibility constraints
if comp_level != "H":
    C16 = stage2.addConstrs((z[r[0], u] + z[r[1], u] <= 1)
                           for inc in INC_dict.keys() for r in INC_dict[inc]
                           for u in list(set(U_r[r[0]]) & set(U_r[r[1]])))

# Fixed aircraft routing
C20 = stage2.addConstrs((x[a, k] == X_tuple_first_stage_dict[(a, k)]
                         for (a, k) in X_tuple_first_stage_dict.keys()), name='C20')


# Fixed request routing
C21 = stage2.addConstrs((z[r, u] <= y[f,u]
                     for r in R_dict.keys() for u in U_r[r]
                         for f in F_r_first_stage[r] if f in F_u[u]),
                      name="C21")
C22 = stage2.addConstrs((z[r, u] + y[f,u] <= 1
                     for r in R_dict.keys() for u in U_r[r]
                         for f in (item for item in F_u[u] if item not in F_r_first_stage[r])),
                      name="C22")



#%%
########################################################################################################################
################################################# OBJECTIVE ############################################################
########################################################################################################################
obj = LinExpr()

# Operational cost due to routing of aircraft fleet
for f in F_dict.keys():
    for k in K_f[f]:
        obj -= K[k]['OC'] * F_info[f]['distance'] * x[f, k]

## Operational cost due to routing of ULDs

for u in U_dict.keys():
    C_ULD = V_info[U_dict[u][2]][5]
    for f in F_u[u]:
        obj -= C_ULD * F_info[f]['distance'] * y[f, u]

# Revenue due to transporting requests
R_avg = 4000  # Euros/ton
for r in R_dict.keys():
    for u in U_r[r]:
        obj += R_avg * R_dict[r][7] * R_dict[r][5] * z[r, u]

# Add a small cost to enter the TSN from the dummy source
# to avoid symmetries
eps = 0.01
for a in Source_TSN_dict.keys():
    for k, v in K.items():
        obj -= eps * x[a, k]

stage2.setObjective(obj, GRB.MAXIMIZE)
stage2.update()

end_time = time.time()

#%%
########################################################################################################################
################################################# SOLVE FIRST STAGE ####################################################
########################################################################################################################
log_path = os.path.join("logs", folder_name+"_SEQ2.log")

print(f'Having {3600 - sequential_stage1.elapsed_time} seconds left for the second stage')

stage2.setParam('MIPGap', 0.001)
stage2.setParam('TimeLimit', (3600) - sequential_stage1.elapsed_time)  # seconds
stage2.setParam("LogFile", log_path)
stage2.optimize()


x_result = []
y_result = []
z_result = []
for var in stage2.getVars():
    if var.Xn > 0.01:
        if var.varName.startswith('x'):
            x_result.append([var.varName, var.Xn])
        if var.varName.startswith('y'):
            y_result.append([var.varName, var.Xn])
        if var.varName.startswith('z'):
            z_result.append([var.varName, var.Xn])

x_dict = {}
y_dict = {}
z_dict = {}
for item in x_result:
    # Extract the indices from the string
    indices_str = item[0][2:-1]  # Remove 'x[' at the start and ']' at the end
    index_tuple = tuple(map(int, indices_str.split(',')))  # Convert to tuple of integers
    x_dict[index_tuple] = round(item[1])
for item in y_result:
    # Extract the indices from the string
    indices_str = item[0][2:-1]  # Remove 'y[' at the start and ']' at the end
    index_tuple = tuple(map(int, indices_str.split(',')))  # Convert to tuple of integers
    y_dict[index_tuple] = round(item[1])
for item in z_result:
    # Extract the indices from the string
    indices_str = item[0][2:-1]  # Remove 'z[' at the start and ']' at the end
    index_tuple = tuple(map(int, indices_str.split(',')))  # Convert to tuple of integers
    z_dict[index_tuple] = round(item[1])


full_folder_path = os.path.join("pickle_files", folder_name+"_SEQ")
os.makedirs(full_folder_path, exist_ok=True)

# Specify the file names
x_dict_file = os.path.join(full_folder_path, 'x_data.pickle')
y_dict_file = os.path.join(full_folder_path, 'y_data.pickle')
z_dict_file = os.path.join(full_folder_path, 'z_data.pickle')

## Write the dictionaries to pickle files
with open(x_dict_file, 'wb') as file:
    pickle.dump(x_dict, file)

with open(y_dict_file, 'wb') as file:
    pickle.dump(y_dict, file)

with open(z_dict_file, 'wb') as file:
    pickle.dump(z_dict, file)

print(f"Dictionaries of x, y, and z values have been written to {x_dict_file}, {y_dict_file}, and {z_dict_file} respectively.")

#print(y_result)
#
#print(z_result)

