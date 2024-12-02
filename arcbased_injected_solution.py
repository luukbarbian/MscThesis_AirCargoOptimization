# -*- coding: utf-8 -*-
"""
Created on Wed April 10 13:13:03 2024

@author: luuk barbian
"""

import pandas as pd
import datetime
import numpy as np
import random
import os
import time
import setsparameters
from gurobipy import Model,GRB,LinExpr,quicksum
import matplotlib.pyplot as plt
import pickle


start_time = time.time()
random.seed(42)
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

#%%
# Solution injection

name = 'full_H_50_restricted_subset15_5it_w_max_cap200_seed42_CG'

# Specify the file names
x_file_name = 'pickle_files/'+name+'/x_data.pickle'
y_file_name = 'pickle_files/'+name+'/y_data.pickle'
z_file_name = 'pickle_files/'+name+'/z_data.pickle'


# Write the dictionaries to pickle files
with open(x_file_name, "rb") as file:
    x_dict = pickle.load(file)

with open(y_file_name, "rb") as file:
    y_dict = pickle.load(file)

with open(z_file_name, "rb") as file:
    z_dict = pickle.load(file)

for (a, k) in x_dict.keys():
    SOL1 = model.addConstr(x[a, k] == x_dict[(a,k)], name='x_sol')
for (a, u) in y_dict.keys():
    SOL2 = model.addConstr(y[a, u] == y_dict[(a,u)], name='y_sol')
for (r, u) in z_dict.keys():
    SOL3 = model.addConstr(z[r, u] == z_dict[(r,u)], name='z_sol')



#%%

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

# Solve
model.setParam('MIPGap', 0.001)
model.setParam('TimeLimit', 1200)  # seconds
model.setParam("LogFile", "arcbased_injection.log")
model.Params.lazyConstraints = 1
model.optimize(callback_lazyConstrs)

# %%

# All the post-processing starts here

solution = []

# Retrieve variable names and values
for v in model.getVars():
    solution.append([v.varName, v.x])

# Function to print constraints and their parameters
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
        print(f"C6[{key}] : {expr_str} <= {constr.RHS}")
def print_constraints(constraints, model):
    for constr in constraints.values():
        print(constr.ConstrName, ":", model.getRow(constr), "<=", constr.RHS)

# Print all constraints and their parameters
print_constraints_with_values(C6, model)

# Retrieve active routing variables
active_variables = []
for i in range(0, len(solution)):
    if solution[i][1] >= 0.01:
        active_variables.append([solution[i][0], solution[i][1]])

# Store active aircraft routing decision variables, meaning these arcs are used by aircraft >= 1
x_variables = []
for v in active_variables:
    if v[0][0] == 'x':
        idx_oprnBrckt = [x for x, char in enumerate(v[0]) if char == '['][0]
        idx_comma = [x for x, char in enumerate(v[0]) if char == ','][0]
        idx_closedBrckt = [x for x, char in enumerate(v[0]) if char == ']'][0]
        x_variables.append((int(v[0][idx_oprnBrckt + 1:idx_comma]),
                            int(v[0][idx_comma + 1:idx_closedBrckt])))

# %%
# Active aircraft routing decision variables dict,
# Key is fleet type, value is the arc that is active
# Note this variable is not binary so one arc can have multiple aircraft of the same fleet using it. This is often the
# case for the source -> sink arc.
x_variables_dict = {k: [a[0] for a in x_variables if a[1] == k] for k in K.keys()}

# Active flight arcs, key is fleet type, value is the active arc
F_k_sol = {k: [x for x in x_variables_dict[k] if x in F_dict.keys()]
           for k in K.keys()}
# Active flight arcs, key is fleet type, value is the active arc in tuple form: (('BOG', 1698802800), ('MIA', 1698812100))
F_k_sol_tuple = {k: [(N_dict_idx_at[F_dict[f][0]], N_dict_idx_at[F_dict[f][1]])
                     for f in F_k_sol[k]]
                 for k in K.keys()}
F_sol_list = [value for sublist in F_k_sol.values() for value in sublist]
Arc_sol_list = [value for sublist in x_variables_dict.values() for value in sublist]
# Active flight arcs, key is fleet type, value is the active arc in tuple form. I guess sorted on time.
F_k_sol_tuple_srt = {k: sorted(F_k_sol_tuple[k], key=lambda x: x[0][1]) for k in K.keys()}
# %%

# Flight arcs tuple
F_tuple = {f: (N_dict_idx_at[n[0]], N_dict_idx_at[n[1]]) for f, n in F_dict.items()}

# Store active ULD routing decision variables, meaning these ULDs use an arc in the TSN
y_variables = []
for v in active_variables:
    if v[0][0] == 'y':
        idx_oprnBrckt = [x for x, char in enumerate(v[0]) if char == '['][0]
        idx_comma = [x for x, char in enumerate(v[0]) if char == ','][0]
        idx_closedBrckt = [x for x, char in enumerate(v[0]) if char == ']'][0]
        y_variables.append((int(v[0][idx_oprnBrckt + 1:idx_comma]),
                            int(v[0][idx_comma + 1:idx_closedBrckt])))

# Active ULD routing decision variables dict,
# Key is ULD id, value is the arc that is active
y_variables_dict = {u: [a[0] for a in y_variables if a[1] == u] for u in U_dict.keys()}

# Active ULD routing decision variables dict for flight arcs,
# Key is ULD id, value is the FLIGHT arc that is active
F_u_sol = {u: [f for f in y_variables_dict[u] if f in F_dict.keys()]
           for u in U_dict.keys()}

# This gives the same as y_variables_dict?? probably did that myself oops
A_u_sol = {u: [a for a in y_variables_dict[u] if a in A_all_dict_3.keys()]
           for u in U_dict.keys()}

# List of ULD ids that are used in the TSN
U_used = [u for u in U_dict.keys() if len(F_u_sol[u]) > 0]

# Active ULD routing decision variables dict for flight arcs,
# Key is flight arc, value is ULD id that uses that flight arc
U_f_sol = {f: [u for u in U_used if f in F_u_sol[u]] for f in F_dict.keys()}

# Active ULD routing decision variables dict,
# Key is the arc, value is ULD id that uses that arc
U_a_sol = {a: [u for u in U_used if a in A_u_sol[u]] for a in A_all_dict_3.keys()}

# Store active request assignment decision variables, meaning these requests get assigned to a ULD
z_variables = []
for v in active_variables:
    if v[0][0] == 'z':
        idx_oprnBrckt = [x for x, char in enumerate(v[0]) if char == '['][0]
        idx_comma = [x for x, char in enumerate(v[0]) if char == ','][0]
        idx_closedBrckt = [x for x, char in enumerate(v[0]) if char == ']'][0]
        z_variables.append((int(v[0][idx_oprnBrckt + 1:idx_comma]),
                            int(v[0][idx_comma + 1:idx_closedBrckt])))

# Active request assignment decision variables dict
# Key is request, value is ULD id that it is assigned to
z_variables_dict = {r: [a[1] for a in z_variables if a[0] == r] for r in R_dict.keys()}

# List of cargo requests that gets transported through the network
R_transported = [r for r in R_dict.keys() if len(z_variables_dict[r]) > 0]

# Active request assignment decision variables dict
# Key is ULD id, value is the requests that are assigned to it
R_u_sol = {u: [r for r in R_transported if z_variables_dict[r][0] == u]
           for u in U_used}

# Dictionary with info on every ULD
U_R_info = {u: {'W': np.round(sum([R_dict[r][5] for r in R_u_sol[u]]), 1),
                'W_max': V_info[U_dict[u][2]][0],
                'V': np.round(sum([R_dict[r][6] for r in R_u_sol[u]]), 1),
                'V_max': V_info[U_dict[u][2]][1]} for u in U_used}

# With Revenue=4,000 we transport 427 requests and use 159 ULDs
# With Revenue=10,000 we transport 811 requests and use 326 ULDs

# %%

# Dict to count how many ULDs of each type use each flight arc, considering 4 ULD types
U_f_count = {f: {k: len([u for u in u_all if U_dict[u][2] == k])
                 for k in V_info.keys()}
             for f, u_all in U_f_sol.items() if len(u_all) != 0}

# Dict to count how many ULDs of each type use each flight arc, considering 2 ULD types
ULD_f_distribution = {f: {k: sum([U_f_count[f][i] for i in idxs])
                          for k, idxs in V_ULD_type_idx_dict.items()} for f, u in U_f_count.items()}

# Dict with key being fight arc, and value being the fleet type that uses that flight arc
F_k_type = {f: [k for k in K.keys() if f in F_k_sol[k]][0] for f in ULD_f_distribution.keys()}

# Dict with key being flight arc, and value indicating maximum number of allowed ULDs per type
ULD_f_max = {f: {uld_type: K[k]['ULD_conf'][idx_uld_type] for idx_uld_type, uld_type in enumerate(V_ULD_type)} for f, k
             in F_k_type.items()}
# %%

# Dict to indicate how many ULDs of each type use each flight arc, considering 4 ULD types
# Outer key is flight arc, inner key is ULD type (considering 4 ULD types)
# Value is a list of ULD ids of that ULD type using that flight arc
U_f_distribution = {f: {k: [u for u in u_all if U_dict[u][2] == k]
                        for k in V_info.keys()}
                    for f, u_all in U_f_sol.items() if len(u_all) != 0}

# Dict to indicate the weight of all ULDs combined using that flight arc
# Key is flight arc, value is combined weight
W_f_distribution = {f: np.round(sum([sum([U_R_info[u]['W'] for u in ulds])
                                     for idx, ulds in v.items()]), 1)
                    for f, v in U_f_distribution.items()}
# Dict to indicate the maximum allowed weight of all ULDs combined using that flight arc
# Key is flight arc, value is maximum allowed combined weight
W_f_max = {f: W_fk[(f, k)] for f, k in F_k_type.items()}
# %%

Revenue_transported = sum(R_avg * [R_dict[r][7] * R_dict[r][5] for r in R_transported])
Revenue_all = sum(R_avg * [R_dict[r][7] * R_dict[r][5] for r in R_dict.keys()])

# %%

# Dict to indicate load factor per flight arc
# Key is flight arc, value is the load factor considering combined ULD weight
LF_f = {f: np.round(W_f_distribution[f] / W_f_max[f] * 100, 1) for f in W_f_distribution.keys()}
print(LF_f)
# %%

# Dict to indicate which requests belong to which OD pair
# Key is OD pair: ('AMS', 'MIA'), key is cargo requests
R_OD = {(o, d): [r for r, v in R_dict.items() if v[0] == o and v[1] == d] for o in airports_IATA for d in airports_IATA if
        d != o}

# Dict to indicate which requests are actually transported per OD pair
# Key is OD pair: ('AMS', 'MIA'), key is cargo requests
R_OD_transported = {od: [req for req in R_OD[od]
                         if req in R_transported]
                    for od in R_OD.keys()}

# Dict to indicate the ratio of transported requests vs not transported requests per OD pair
# Key is OD pair: ('AMS', 'MIA'), key is % of transported requests, if no requests exist for OD pair ratio == -1
R_OD_ratio = {od: np.round(len(R_OD_transported[od]) / len(R_OD[od]) * 100, 1) if
len(R_OD[od]) > 0 else -1
              for od in R_OD.keys()}
print(R_OD_ratio)
# %%
import pickle

#with open(os.path.join(cwd, 'LF_f.pickle'), 'wb') as handle:
#    pickle.dump(LF_f, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('LF_f.pickle', 'rb') as handle:
#     test = pickle.load(handle)

# %%

# Determine connectivity of the network, i.e., if we can reach every
# potential OD pair with the set of flights we can operate

import networkx as nx

G_nodes = []
for k, v in G_dict.items():
    G_nodes.append(v[0])
    G_nodes.append(v[1])
for k, v in F_dict.items():
    G_nodes.append(v[0])
    G_nodes.append(v[1])

G_nodes = list(np.unique(G_nodes))

G = nx.DiGraph()

for n in G_nodes:
    G.add_node(n)

for k, v in G_dict.items():
    G.add_edge(v[0], v[1], weight=1)

for k, v in F_dict.items():
    G.add_edge(v[0], v[1], weight=1)

# Node index for the source node of each airport
source_node_airports = {k: N_dict_at_idx[v] for k, v
                        in earliest_node_airport.items()}
# Node index for the sink node of each airport
target_node_airports = {k: N_dict_at_idx[v] for k, v
                        in latest_node_airport.items()}

# Dictionary indicating the shortest path per OD pair
# Key is OD pair, value is the shortest path possible, if no path possible value is -1
SP_dict = {}
for o, n_o in source_node_airports.items():
    for d, n_d in target_node_airports.items():
        if d != o:
            try:
                path = nx.shortest_path(G, source=n_o, target=n_d)
                SP_dict[(o, d)] = path
            except (nx.NetworkXNoPath, KeyError):
                SP_dict[(o, d)] = [-1]

# List of requests that can possibly be transported due to network layout
R_transportable = [r for r, v in R_dict.items()
                   if SP_dict[(v[0], v[1])][0] != -1]

# %%

# Check that no incompatible items are assigned to the same ULD

if comp_level != "H":
    for inc_types in INC_dict.keys():
        for r_inc in INC_dict[inc_types]:
            for u in list(set(U_r[r_inc[0]]) & set(U_r[r_inc[1]])):
                if u in U_used and (r_inc[0] in R_u_sol[u] and r_inc[1] in R_u_sol[u]):
                    print('Issue with ULD %i and cargo requests %i and %i' % (u, r[0], r[1]))

# %%

#print("Nodes", N_dict)
#
#print("Ground arcs", G_dict)
#print("Flight arcs", F_dict)
#print("Source -> TSN arcs", Source_TSN_dict)
#print("TSN -> Sink arcs", TSN_Sink_dict)
#print("Source -> Sink arcs", Source_Sink_dict)
#print("ULD Source -> TSN arcs", ULD_source_TSN_dict)
#print("TSN -> ULD Sink arcs", TSN_ULD_sink_dict)
#
#print("Fleet",K)
#print(len(U_dict),"ULDs")
#print(len(R),"Cargo requests", R)

print(x_variables)
print(x_variables_dict)
#
print(y_variables)
print(y_variables_dict)
print(U_used)
print(U_f_sol)
print(U_a_sol)

print(z_variables)
print(z_variables_dict)
#
print(Revenue_transported)
#for key in x.keys():
#    print(f"x_ak[{key}] = {x[key].x}")
#%%
airport_names = []
times = []
plt.figure(figsize=(10, 6))
for idx, (airport, time) in N_dict.items():
    if airport not in ['Source', 'Sink']:
        airport_names.append(airport)
        times.append(time)

# Plotting
plt.scatter(times, airport_names, color='blue', marker='o')
plt.xlabel('Time')
plt.ylabel('Airport')
plt.title('Airport vs. Time Scatter Plot')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

for idx, (start, end) in G_dict.items():
    if idx in Arc_sol_list:
        plt.plot([N_dict[start][1], N_dict[end][1]], [N_dict[start][0], N_dict[end][0]], color='black', linewidth=2.0)
    else:
        plt.plot([N_dict[start][1], N_dict[end][1]], [N_dict[start][0], N_dict[end][0]], color='black', linestyle='--', linewidth=0.8)

for idx, (start, end) in F_dict.items():
    if idx in F_sol_list:
        plt.plot([N_dict[start][1], N_dict[end][1]], [N_dict[start][0], N_dict[end][0]], color='orange', linewidth=2.0)
    else:
        plt.plot([N_dict[start][1], N_dict[end][1]], [N_dict[start][0], N_dict[end][0]], color='orange', linestyle='--', linewidth=0.8)
plt.scatter(times, airport_names, color='blue', marker='o')

plt.show()
