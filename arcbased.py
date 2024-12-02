# -*- coding: utf-8 -*-
"""
Created on Wed April 10 13:13:03 2024

@author: luuk barbian
"""

import random
import os
import time
import setsparameters
from gurobipy import Model,GRB,LinExpr,quicksum
import matplotlib.pyplot as plt
import pickle


start_time = time.time()
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

schedule_small = setsparameters.schedule_small
clustering_requests = setsparameters.clustering_requests
clustering_maximum = setsparameters.clustering_maximum
subset = setsparameters.subset
max_num_of_ulds_per_R = setsparameters.max_num_of_ulds_per_R
comp_level = setsparameters.comp_level
Payload = setsparameters.Payload
free_aircraft_placement = setsparameters.free_aircraft_placement


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
############################################### DEFINING THE MODEL #####################################################
########################################################################################################################

# Setup model
model = Model()

# Decision variables

x = {}  # aircraft fleet routing decision variables
y = {}  # ULD routing decision variables
z = {}  # Request to ULD assignment decision variables
w = {}  # y*w linearization decision variables

# Aircraft routing decision variables for ground arcs (integer)
for g in G_dict.keys():
    for k, v in K.items():
        x[g, k] = model.addVar(lb=0, ub=v['#'], vtype=GRB.INTEGER, name='x[%s,%s]' % (g, k))

# Aircraft routing decision variables for flight arcs (binary)
for f in F_dict.keys():
    for k in K_f[f]:
        x[f, k] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='x[%s,%s]' % (f, k))

# Aircraft routing decision variables from dummy source to TSN (integer)
for a in Source_TSN_dict.keys():
    for k, v in K.items():
        x[a, k] = model.addVar(lb=0, ub=v['#'], vtype=GRB.INTEGER, name='x[%s,%s]' % (a, k))

# Aircraft routing decision variables from TSN to dummy sink (integer)
for a in TSN_Sink_dict.keys():
    for k, v in K.items():
        x[a, k] = model.addVar(lb=0, ub=v['#'], vtype=GRB.INTEGER, name='x[%s,%s]' % (a, k))

# Aircraft routing decision variable from dummy source to dummy sink (integer)
for k, v in K.items():
    for a in Source_Sink_dict.keys():
        x[a, k] = model.addVar(lb=0, ub=v['#'], vtype=GRB.INTEGER, name='x[%s,%s]' % (a, k))

# ULD routing decision variables from source node to TSN node
for idx, a in enumerate(ULD_source_TSN_dict.keys()):
    y[a, idx] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='y[%s,%s]' % (a, idx))

# ULD routing decision variables from TSN node to sink node
for idx, a in enumerate(TSN_ULD_sink_dict.keys()):
    y[a, idx] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='y[%s,%s]' % (a, idx))

# ULD routing decision variables for ground arcs
for u in U_dict.keys():
    for g in G_u[u]:
        y[g, u] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='y[%s,%s]' % (g, u))

# ULD routing decision variables for flight arcs
for u in U_dict.keys():
    for f in F_u[u]:
        y[f, u] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='y[%s,%s]' % (f, u))

# Request to ULD assignment decision variables
for r in R_dict.keys():
    for u in U_r[r]:
        z[r, u] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='z[%s,%s]' % (r, u))

# Linearization of y*z decision variables
for r in R_dict.keys():
    for u in U_r[r]:
        for f in F_u[u]:
            w[f, u, r] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='w[%s,%s,%s]' % (f, u, r))

#%%
########################################################################################################################
############################################### CONSTRAINTS ############################################################
########################################################################################################################

# Every flight arc can be flown at most by one aircraft type
# This should be an equality constraint according to the manuscript meaning each flight would need to have an
# aircraft type assigned, with <= some flight arcs can be left 'unflown'
if free_aircraft_placement:
    C1 = model.addConstrs((quicksum(x[f, k] for k in K_f[f]) <= 1 for f in F_dict.keys()),
                          name="C1")
if not free_aircraft_placement:
    C1 = model.addConstrs((quicksum(x[f, k] for k in K_f[f]) == 1 for f in F_dict.keys()),
                          name="C1")

# For each flight leg, we can carry as many ULDs of type v as there are allowed by the configuration
C2 = model.addConstrs((quicksum(y[f, u] for u in U_v_f[(ULD_type_idx, f)]) -
                       quicksum(N_kv[k][ULD_type_idx] * x[f, k] for k in K_f[f]) <= 0
                       for f in F_dict.keys() for ULD_type_idx in U_uld_type_idx.keys()),
                      name="C2")

# For each ULD, we must satisfy the weight capacity
C3 = model.addConstrs((quicksum(R_dict[r][5] * z[r, u] for r in R_u[u]) <=
                       V_info[U_dict[u][2]][0] for u in U_dict.keys()),
                      name="C3")

# For each ULD, we must satisfy the volume capacity
C4 = model.addConstrs((quicksum(R_dict[r][6] * z[r, u] for r in R_u[u]) <=
                       V_info[U_dict[u][2]][1] for u in U_dict.keys()),
                      name="C4")

# Each request can be assigned to at most one ULD
C5 = model.addConstrs((quicksum(z[r, u] for u in U_r[r]) <=
                       1 for r in R_dict.keys()),
                      name="C5")

# The overall number of aircraft of a specific type must leave the dummy source
C6 = model.addConstrs(((quicksum(x[a, k] for a in Source_TSN_dict.keys()) + \
                        quicksum(x[a, k] for a in Source_Sink_dict.keys()) ==
                        N_k[k] for k in K.keys())), name="C6")

# The overall number of aircraft of a specific type must arrive to the dummy sink
C7 = model.addConstrs(((quicksum(x[a, k] for a in TSN_Sink_dict.keys()) + \
                        quicksum(x[a, k] for a in Source_Sink_dict.keys()) ==
                        N_k[k] for k in K.keys())), name="C7")

# Conservation of flow in the TSN per aircraft type
a: object
C8 = model.addConstrs((quicksum(x[a, k] for a in AK_k_n_plus[(k, n)]) -
                       quicksum(x[a, k] for a in AK_k_n_minus[(k, n)]) == 0
                       for k in K.keys() for n in N_K_realTSN.keys()), name="C8")

# Conservation of flow in the TSN per ULD
C9 = model.addConstrs((quicksum(y[a, u] for a in AU_u_n_plus[(u, n)]) -
                       quicksum(y[a, u] for a in AU_u_n_minus[(u, n)]) == 0
                       for u in U_dict.keys() for n in N_K_realTSN.keys()), name="C9")

C14 = model.addConstrs((z[r, u] -
                        quicksum(y[f, u] for f in F_r_plus[r, u]) <= 0
                        for r in R_dict.keys() for u in U_r[r]), name="C14")

C15 = model.addConstrs((z[r, u] -
                        quicksum(y[f, u] for f in F_r_minus[r, u]) <= 0
                        for r in R_dict.keys() for u in U_r[r]), name="C15")

# Incompatibility constraints
if comp_level != "H":
    C16 = model.addConstrs((z[r[0], u] + z[r[1], u] <= 1)
                           for inc in INC_dict.keys() for r in INC_dict[inc]
                           for u in list(set(U_r[r[0]]) & set(U_r[r[1]])))
#################################
### Define objective function ###
#################################
obj = LinExpr()

# Operational cost due to routing of aircraft fleet
for f in F_dict.keys():
    for k in K_f[f]:
        obj -= K[k]['OC'] * F_info[f]['distance'] * x[f, k]

# Operational cost due to routing of ULDs
#C_ULD = .1
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

model.setObjective(obj, GRB.MAXIMIZE)
model.update()

end_time = time.time()


# %%

# Defining callback function
def callback_lazyConstrs(model, where):
    if where == GRB.Callback.MIPSOL:
        # Get the current solution
        x_val = model.cbGetSolution(x)
        y_val = model.cbGetSolution(y)
        z_val = model.cbGetSolution(z)
        print('Checking for lazy constraint...')
        fk_used = [k for k, v in x_val.items() if v >= 0.99 and k[0] in list(F_dict.keys())]
        # For every used arc, check if payload capacity is not exceeded
        for fk in fk_used:
            f = fk[0]  # flight arc
            k = fk[1]  # aircraft type
            u_f = [k[1] for k, v in y_val.items() if v >= 0.99 and k[0] == f]  # All ULDs using that flight arc
            r_u = {u: [k[0] for k, v in z_val.items() if v >= 0.99 and k[1] == u] for u in u_f}
            Payload = sum([R_dict[k[0]][5] for k, v in z_val.items()
                           if v >= 0.99 and k[1] in u_f])  # Weight of all requests carried along that flght arc
            # If payload capacity is exceeded, add lazy constraint
            if Payload > W_fk[(f, k)]:
                print('Lazy constraint is added!')

                for u in u_f:
                    for r in r_u[u]:
                        model.cbLazy(y[f, u] + z[r, u] - w[f, u, r] <= 1)

                for u in u_f:
                    for r in r_u[u]:
                        model.cbLazy(w[f, u, r] - y[f, u] <= 0)

                for u in u_f:
                    for r in r_u[u]:
                        model.cbLazy(w[f, u, r] - z[r, u] <= 0)

                model.cbLazy(quicksum(R_dict[r][5] * w[f, u, r] for u in u_f for r in r_u[u]) -
                             W_fk[(f, k)] * x[f, k] <= 0)


# %%

log_path = os.path.join("logs", folder_name+"_AB.log")


# Solve
model.setParam('MIPGap', 0.001)
model.setParam('TimeLimit', 3600)  # seconds
model.setParam("LogFile", log_path)
model.Params.lazyConstraints = 1
model.optimize(callback_lazyConstrs)

#%%
x_result = []
y_result = []
z_result = []
for var in model.getVars():
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


full_folder_path = os.path.join("pickle_files", folder_name+"_12hours_AB")
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

# %%
