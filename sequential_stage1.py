# -*- coding: utf-8 -*-
"""
Created on Wed July 22 16:33:21 2024

@author: luuk barbian
"""

import random
import os
import time
import setsparameters
from gurobipy import Model,GRB,LinExpr,quicksum
import numpy as np
import logging

start_time = time.time()
print('Starting the clock')
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
#G_u = setsparameters.G_u
#F_u = setsparameters.F_u
U_v_f = setsparameters.U_v_f
AK_k_n_plus = setsparameters.AK_k_n_plus
AK_k_n_minus = setsparameters.AK_k_n_minus
N_K_realTSN = setsparameters.N_K_realTSN
#AU_u_n_plus = setsparameters.AU_u_n_plus
#AU_u_n_minus = setsparameters.AU_u_n_minus
F_info = setsparameters.F_info
W_fk = setsparameters.W_fk
earliest_node_airport = setsparameters.earliest_node_airport
latest_node_airport = setsparameters.latest_node_airport
#F_r_plus = setsparameters.F_r_plus
#F_r_minus = setsparameters.F_r_minus


U_dict = setsparameters.U_dict
#U_r = setsparameters.U_r
R_u = setsparameters.R_u
U_uld_type_idx = setsparameters.U_uld_type_idx
V_info = setsparameters.V_info
V_ULD_type_idx_dict = setsparameters.V_ULD_type_idx_dict
V_ULD_type = setsparameters.V_ULD_type

R_dict = setsparameters.R_dict

dummy_U_dict = setsparameters.dummy_U_dict
dummy_U_r = setsparameters.dummy_U_r
dummy_U_f = setsparameters.dummy_U_f
dummy_AU_u_n_plus = setsparameters.dummy_AU_u_n_plus
dummy_AU_u_n_minus = setsparameters.dummy_AU_u_n_minus
dummy_ULD_source_TSN_dict = setsparameters.dummy_ULD_source_TSN_dict
dummy_TSN_ULD_sink_dict = setsparameters.dummy_TSN_ULD_sink_dict
dummy_G_u = setsparameters.dummy_G_u
dummy_F_u = setsparameters.dummy_F_u
dummy_F_r_plus = setsparameters.dummy_F_r_plus
dummy_F_r_minus = setsparameters.dummy_F_r_minus
V_info_simp = setsparameters.V_info_simp

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
############################################# DEFINING THE MODEL FIRST STAGE ###########################################
########################################################################################################################

# Setup model
stage1 = Model()

# Decision variables

x = {}  # aircraft fleet routing decision variables
y = {}  # ULD routing decision variables
z = {}  # Request to ULD assignment decision variables

# Aircraft routing decision variables for ground arcs (integer)
for g in G_dict.keys():
    for k, v in K.items():
        x[g, k] = stage1.addVar(lb=0, ub=v['#'], vtype=GRB.INTEGER, name='x[%s,%s]' % (g, k))

# Aircraft routing decision variables for flight arcs (binary)
for f in F_dict.keys():
    for k in K_f[f]:
        x[f, k] = stage1.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='x[%s,%s]' % (f, k))

# Aircraft routing decision variables from dummy source to TSN (integer)
for a in Source_TSN_dict.keys():
    for k, v in K.items():
        x[a, k] = stage1.addVar(lb=0, ub=v['#'], vtype=GRB.INTEGER, name='x[%s,%s]' % (a, k))

# Aircraft routing decision variables from TSN to dummy sink (integer)
for a in TSN_Sink_dict.keys():
    for k, v in K.items():
        x[a, k] = stage1.addVar(lb=0, ub=v['#'], vtype=GRB.INTEGER, name='x[%s,%s]' % (a, k))

# Aircraft routing decision variable from dummy source to dummy sink (integer)
for k, v in K.items():
    for a in Source_Sink_dict.keys():
        x[a, k] = stage1.addVar(lb=0, ub=v['#'], vtype=GRB.INTEGER, name='x[%s,%s]' % (a, k))

# dummy ULD routing decision variables from source node to TSN node
for idx, a in enumerate(dummy_ULD_source_TSN_dict.keys()):
    y[a, idx] = stage1.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='y[%s,%s]' % (a, idx))

# dummy ULD routing decision variables from TSN node to sink node
for idx, a in enumerate(dummy_TSN_ULD_sink_dict.keys()):
    y[a, idx] = stage1.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='y[%s,%s]' % (a, idx))

# dummy ULD routing decision variables for ground arcs
for u in dummy_U_dict.keys():
    for g in dummy_G_u[u]:
        y[g, u] = stage1.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='y[%s,%s]' % (g, u))

# dummy ULD routing decision variables for flight arcs
for u in dummy_U_dict.keys():
    for f in dummy_F_u[u]:
        y[f, u] = stage1.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='y[%s,%s]' % (f, u))

# Request to dummy ULD assignment decision variables
for r in R_dict.keys():
    for u in dummy_U_r[r]:
        z[r, u] = stage1.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='z[%s,%s]' % (r, u))

#%%
########################################################################################################################
############################################### CONSTRAINTS ############################################################
########################################################################################################################

# Every flight arc can be flown at most by one aircraft type
# This should be an equality constraint according to the manuscript meaning each flight would need to have an
# aircraft type assigned, with <= some flight arcs can be left 'unflown'

if free_aircraft_placement:
    C1 = stage1.addConstrs((quicksum(x[f, k] for k in K_f[f]) <= 1 for f in F_dict.keys()),
                      name="C1")
if not free_aircraft_placement:
    C1 = stage1.addConstrs((quicksum(x[f, k] for k in K_f[f]) == 1 for f in F_dict.keys()),
                           name="C1")

## For each flight leg, we can carry as many ULDs of type v as there are allowed by the configuration
#C2 = model.addConstrs((quicksum(y[f, u] for u in U_v_f[(ULD_type_idx, f)]) -
#                       quicksum(N_kv[k][ULD_type_idx] * x[f, k] for k in K_f[f]) <= 0
#                       for f in F_dict.keys() for ULD_type_idx in U_uld_type_idx.keys()),
#                      name="C2")
#
## For each ULD, we must satisfy the weight capacity
#C3 = model.addConstrs((quicksum(R_dict[r][5] * z[r, u] for r in R_u[u]) <=
#                       V_info[U_dict[u][2]][0] for u in U_dict.keys()),
#                      name="C3")
#
## For each ULD, we must satisfy the volume capacity
#C4 = model.addConstrs((quicksum(R_dict[r][6] * z[r, u] for r in R_u[u]) <=
#                       V_info[U_dict[u][2]][1] for u in U_dict.keys()),
#                      name="C4")
#
# Each request can be assigned to at most one ULD
C5 = stage1.addConstrs((quicksum(z[r, u] for u in dummy_U_r[r]) <=
                       1 for r in R_dict.keys()),
                      name="C5")

# The overall number of aircraft of a specific type must leave the dummy source
C6 = stage1.addConstrs(((quicksum(x[a, k] for a in Source_TSN_dict.keys()) + \
                        quicksum(x[a, k] for a in Source_Sink_dict.keys()) ==
                        N_k[k] for k in K.keys())), name="C6")

# The overall number of aircraft of a specific type must arrive to the dummy sink
C7 = stage1.addConstrs(((quicksum(x[a, k] for a in TSN_Sink_dict.keys()) + \
                        quicksum(x[a, k] for a in Source_Sink_dict.keys()) ==
                        N_k[k] for k in K.keys())), name="C7")

# Conservation of flow in the TSN per aircraft type
C8 = stage1.addConstrs((quicksum(x[a, k] for a in AK_k_n_plus[(k, n)]) -
                       quicksum(x[a, k] for a in AK_k_n_minus[(k, n)]) == 0
                       for k in K.keys() for n in N_K_realTSN.keys()), name="C8")

# Conservation of flow in the TSN per ULD
C9 = stage1.addConstrs((quicksum(y[a, u] for a in dummy_AU_u_n_plus[(u, n)]) -
                       quicksum(y[a, u] for a in dummy_AU_u_n_minus[(u, n)]) == 0
                       for u in dummy_U_dict.keys() for n in N_K_realTSN.keys()), name="C9")


# For each flight leg, we have a weight constraint by the ULDs of type v as there are allowed by the configuration
C10 = stage1.addConstrs((quicksum(dummy_U_dict[u]['Weight'] * y[f, u] for u in dummy_U_f[f]) -
                       quicksum(N_kv[k][ULD_type_idx] * V_info_simp[ULD_type_idx]['Weight']* x[f, k]
                                for k in K_f[f] for ULD_type_idx in U_uld_type_idx.keys()) <= 0
                       for f in F_dict.keys()),
                      name="C10")

# For each flight leg, we have a volume constraint by the ULDs of type v as there are allowed by the configuration
C11 = stage1.addConstrs((quicksum(dummy_U_dict[u]['Volume'] * y[f, u] for u in dummy_U_f[f]) -
                       quicksum(N_kv[k][ULD_type_idx] * V_info_simp[ULD_type_idx]['Volume']* x[f, k]
                                for k in K_f[f] for ULD_type_idx in U_uld_type_idx.keys()) <= 0
                       for f in F_dict.keys()),
                      name="C11")

# For each flight leg, we have a weight constraint based on the payload range diagram
C12 = stage1.addConstrs((quicksum(dummy_U_dict[u]['Weight'] * y[f, u] for u in dummy_U_f[f]) -
                       quicksum(W_fk[(f, k)] * x[f, k] for k in K_f[f]) <= 0
                       for f in F_dict.keys()),
                      name="C12")


C14 = stage1.addConstrs((z[r, u] -
                        quicksum(y[f, u] for f in dummy_F_r_plus[r, u]) <= 0
                        for r in R_dict.keys() for u in dummy_U_r[r]), name="C14")

C15 = stage1.addConstrs((z[r, u] -
                        quicksum(y[f, u] for f in dummy_F_r_minus[r, u]) <= 0
                        for r in R_dict.keys() for u in dummy_U_r[r]), name="C15")

#%%
########################################################################################################################
################################################# OBJECTIVE ############################################################
########################################################################################################################
obj = LinExpr()

# Operational cost due to routing of aircraft fleet
for f in F_dict.keys():
    for k in K_f[f]:
        obj -= K[k]['OC'] * F_info[f]['distance'] * x[f, k]

### Operational cost due to routing of ULDs
#C_ULD = .1
#for u in dummy_U_dict.keys():
#    for f in dummy_F_u[u]:
#        obj -= C_ULD * F_info[f]['distance'] * y[f, u]

# Revenue due to transporting requests
R_avg = 4000  # Euros/ton
for r in R_dict.keys():
    for u in dummy_U_r[r]:
        obj += R_avg * R_dict[r][7] * R_dict[r][5] * z[r, u]

# Add a small cost to enter the TSN from the dummy source
# to avoid symmetries
eps = 0.01
for a in Source_TSN_dict.keys():
    for k, v in K.items():
        obj -= eps * x[a, k]

stage1.setObjective(obj, GRB.MAXIMIZE)
stage1.update()



#%%
########################################################################################################################
################################################# SOLVE FIRST STAGE ####################################################
########################################################################################################################
log_path = os.path.join("logs", folder_name+"_SEQ1.log")

stage1.setParam('MIPGap', 0.001)
stage1.setParam('TimeLimit', 2 * 3600)  # seconds
stage1.setParam("LogFile", log_path)
stage1.optimize()

end_time = time.time()
elapsed_time = end_time - start_time
print(f'Time elsapsed: {elapsed_time}')


### Configure logging
#logging.basicConfig(
#    level=logging.INFO,  # Log messages with level INFO or higher
#    format='%(asctime)s - %(levelname)s - %(message)s',  # Format of log messages
#    handlers=[
#        logging.FileHandler("logs/solution_analyser/"+folder_name+"_SEQ1_solution_extra.log"),  # Write logs to a file named output.log
#        logging.StreamHandler()  # Also output logs to the console
#    ]
#)
#logging.info(f'Time elsapsed: {elapsed_time}')
#%%
########################################################################################################################
########################################### PROCESS RESULTS FIRST STAGE ################################################
########################################################################################################################
x_result = []
y_result = []
z_result = []
for var in stage1.getVars():
    if var.Xn > 0.01:
        if var.varName.startswith('x'):
            x_result.append([var.varName, var.Xn])
        if var.varName.startswith('y'):
            y_result.append([var.varName, var.Xn])
        if var.varName.startswith('z'):
            z_result.append([var.varName, var.Xn])

X_tuple_first_stage_dict = {}
for item in x_result:
    # Extract the indices from the string
    indices_str = item[0][2:-1]  # Remove 'x[' at the start and ']' at the end
    index_tuple = tuple(map(int, indices_str.split(',')))  # Convert to tuple of integers
    X_tuple_first_stage_dict[index_tuple] = round(item[1])

Arc_r_first_stage_dict = {}
F_r_first_stage_dict = {}

for r in R_dict.keys():
    F_r_first_stage_dict[r] = []

for item in y_result:
    # Extract the first and second numbers from the string index
    index_str = item[0]
    arc, r = map(int, index_str[2:-1].split(','))

    # Add the first number to the list corresponding to the second number in the dictionary
    if r not in Arc_r_first_stage_dict:
        Arc_r_first_stage_dict[r] = []
    Arc_r_first_stage_dict[r].append(arc)

    if arc in F_dict:
        F_r_first_stage_dict[r].append(arc)

#print(X_tuple_first_stage_dict)


#%%
########################################################################################################################
############################################ DEBUG PURPOSES ############################################################
########################################################################################################################

slacks = {}
for c in stage1.getConstrs():
    slacks[c.ConstrName] = c.Slack

C10_slacks = {}
C11_slacks = {}
C12_slacks = {}

# Print dual variables
for constraint_name, slack_value in slacks.items():
    # if dual_value > 0:
    # print(f"Dual variable for constraint {constraint_name}: {dual_value}")
    # Assign dual values for constraint C5 corresponding to gamma_fv
    if constraint_name.startswith('C10'):
        indices = constraint_name.split('[')[1][:-1]
        indices = tuple(map(int, indices.split(',')))
        C10_slacks[indices] = slack_value
    # Assign dual values for constraint C6 corresponding to delta_f
    if constraint_name.startswith('C11'):
        indices = int(constraint_name.split('[')[1][:-1])
        C11_slacks[indices] = slack_value
    ## Assign dual values for constraint C7 corresponding to eta_u
    if constraint_name.startswith('C12'):
        indices = int(constraint_name.split('[')[1][:-1])
        C12_slacks[indices] = slack_value

#print(C10_slacks)
#print(C11_slacks)
#print(C12_slacks)

def print_constraints_with_values(constraints, model):
    for key, constr in constraints.items():
        constr_expr = model.getRow(constr)
        expr_str = ""
        for i in range(constr_expr.size()):
            var = constr_expr.getVar(i)
            coeff = constr_expr.getCoeff(i)
            var_value = var.X
            expr_str += f"{coeff} * {var.VarName}({var_value}) + "
        expr_str = expr_str[:-3]  # Remove the last ' + '

        print(f"C9[{key}] : {expr_str} <= {constr.RHS}")

#print_constraints_with_values(C9, stage1)

#%%
########################################################################################################################
############################################ ANALYSIS LOG ##############################################################
########################################################################################################################

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

flight_costs = 0
Weight_ULD_req = 32 * 1.5 + 30 * 4.6
Vol_ULD_req = 32 * 4.6 + 30 * 10.7
F_uld_lf = {}
F_uld_vf = {}
F_plr_lf = {}

for f in F_dict.keys():
    found = False  # To track if any k exists for the current f
    for k in K.keys():
        if x_dict.get((f, k)) is not None and x_dict.get((f, k)) >= 0.9:

            weight = 0
            volume = 0

            if f == 82:
                logging.info(f'Cargo and ULDs on flight arc 82:')
            for r in R_dict.keys():
                if y_dict.get((f,r)) is not None and y_dict.get((f, r)) >= 0.9:
                    weight += R_dict[r][5]
                    volume += R_dict[r][6]
                    if f == 82:
                        logging.info(f'Request: {r, R_dict[r]}')

            F_uld_lf[f] = weight / Weight_ULD_req
            F_uld_vf[f] = volume / Vol_ULD_req
            F_plr_lf[f] = weight / W_fk[(f, k)]


            print(f'Flight arc {f} is being flown by aricraft type {k} and '
                  f'has a ULD based load factor of: {F_uld_lf[f]} a ULD based volume factor of: {F_uld_vf[f]}'
                  f'and a payload range based load factor of {F_plr_lf[f]}')
            logging.info(f'Flight arc {f} is being flown by aricraft type {k} and '
                  f'has a ULD based load factor of: {F_uld_lf[f]} a ULD based volume factor of: {F_uld_vf[f]}'
                  f'and a payload range based load factor of {F_plr_lf[f]}')


            #print(f'Flight arc {f} is being flown by aircraft type {k}')
            #logging.info(f'Flight arc {f} is being flown by aircraft type {k}')

            flight_costs -= K[k]['OC'] * F_info[f]['distance']
            found = True
    if not found:
        print(f'FLIGHT ARC {f} IS NOT BEING FLOWN BY ANY AIRCRAFT TYPE')
        logging.info(f'FLIGHT ARC {f} IS NOT BEING FLOWN BY ANY AIRCRAFT TYPE')

print(f'Average ULD based load factor over all flights: {sum(F_uld_lf.values()) / len(F_uld_lf)}')
print(f'Average ULD based volume factor over all flights: {sum(F_uld_vf.values()) / len(F_uld_vf)}')
print(f'Average payload range based load factor over all flights: {sum(F_plr_lf.values()) / len(F_plr_lf)}')
print(f'Number of flights: {len(F_plr_lf)}')
print(f'Number of flights where the ULD lf is limiting: {sum(1 for value in F_uld_lf.values() if value > 0.99)}')
print(f'Number of flights where the ULD vf is limiting: {sum(1 for value in F_uld_vf.values() if value > 0.99)}')
print(f'Number of flights where the plr lf is limiting: {sum(1 for value in F_plr_lf.values() if value > 0.99)}')

logging.info(f'Average ULD based load factor over all flights: {sum(F_uld_lf.values()) / len(F_uld_lf)}')
logging.info(f'Average ULD based volume factor over all flights: {sum(F_uld_vf.values()) / len(F_uld_vf)}')
logging.info(f'Average payload range based load factor over all flights: {sum(F_plr_lf.values()) / len(F_plr_lf)}')
logging.info(f'Number of flights: {len(F_plr_lf)}')
logging.info(f'Number of flights where the ULD lf is limiting: {sum(1 for value in F_uld_lf.values() if value > 0.995)}')
logging.info(f'Number of flights where the ULD vf is limiting: {sum(1 for value in F_uld_vf.values() if value > 0.995)}')
logging.info(f'Number of flights where the plr lf is limiting: {sum(1 for value in F_plr_lf.values() if value > 0.995)}')

R_transported = [r for r in R_dict.keys() for u in U_dict.keys() if z_dict.get((r,u)) is not None and z_dict.get((r, u)) >= 0.9]
R_transported_ratio = len(R_transported)/len(R_dict)*100
R_OD = {(o,d):[r for r,v in R_dict.items() if v[0]==o and v[1]==d] for o in airports_IATA for d in airports_IATA if d!=o}
R_OD_transported = {od:[req for req in R_OD[od]
                        if req in R_transported]
                    for od in R_OD.keys()}
R_OD_unfulfilled = {od:[req for req in R_OD[od]
                        if req not in R_transported]
                    for od in R_OD.keys()}
R_OD_ratio = {od: np.round(len(R_OD_transported[od])/len(R_OD[od])*100,1) if
              len(R_OD[od]) > 0 else -1
                    for od in R_OD.keys()}

R_unfulfilled = [r for r in R_dict if r not in R_transported]

revenue = 0
for r in R_transported:
    revenue += R_dict[r][5] * R_dict[r][7] * 4000
    #print('Revenue gained because item transported:',R_dict[r][5] * R_dict[r][7] * 4000)

lost_revenue = 0
for r in R_unfulfilled:
    lost_revenue += R_dict[r][5] * R_dict[r][7] * 4000
    #print('Revenue lost because item NOT transported:',R_dict[r][5] * R_dict[r][7] * 4000)
average_rev_transported = revenue/len(R_transported)
average_rev_unfulfilled = lost_revenue/len(R_unfulfilled)

print(f'Revenue with transported requests: {revenue}')
logging.info(f'Revenue with transported requests: {revenue}')

print(f'Number of transported requests: {len(R_transported)}')
print(f'Number of total requests: {len(R_dict)}')
print(f'Average revenue of transported requests: {average_rev_transported}')
print(f'Average revenue of unfulfilled requests: {average_rev_unfulfilled}')

logging.info(f'Number of transported requests: {len(R_transported)}')
logging.info(f'Number of total requests: {len(R_dict)}')
logging.info(f'Average revenue of transported requests: {average_rev_transported}')
logging.info(f'Average revenue of unfulfilled requests: {average_rev_unfulfilled}')

print(f'Total ratio of requests transported: {R_transported_ratio}')
print(f'Ratio of fulfilled requests per OD pair {R_OD_ratio}')

logging.info(f'Total ratio of requests transported: {R_transported_ratio}')
logging.info(f'Ratio of fulfilled requests per OD pair {R_OD_ratio}')
