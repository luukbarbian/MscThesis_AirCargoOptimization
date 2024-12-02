"""
Created on Wed July 16 15:16:42 2024

@author: luuk barbian
"""

import numpy as np
import pickle
import random
import os
import time
import setsparameters
import inputs
from gurobipy import Model, GRB, LinExpr, quicksum, Column
import matplotlib.pyplot as plt
import networkx as nx
import sys

print(f"Running RMP_CG_classes.py with Python interpreter: {sys.executable}")


linearize_PP = inputs.linearize_PP
w_max = inputs.w_max
w_min = inputs.w_min
number_of_iterations = inputs.number_of_iterations
paths_per_request_cap = inputs.paths_per_request_cap
max_num_paths_per_r = inputs.max_num_paths_per_r
number_of_solutions = inputs.number_of_solutions
number_of_solutions_div = inputs.number_of_solutions_div
diversification = inputs.diversification
div_method = inputs.div_method

if linearize_PP:
    if w_max or w_min:
        raise ValueError("w_max and w_min should be false when adding linearized PP")
    else:
        print(f'TAKING W_UP = W_UP. ADDING LINEARIZATION CONSTRAINTS!! WATCH OUT FOR MEMORY USAGE!')
else:
    if w_max and w_min:
        raise ValueError("w_max and w_min cannot both be true")
    if w_max:
        print(f'NO LINEARIZATION CONSTRAINTS! TAKING W_UP = W_MAX')
    if w_min:
        print(f'NO LINEARIZATION CONSTRAINTS! TAKING W_UP = W_MIN')
    if not w_max and not w_min:
        raise ValueError("either w_max or w_min must be true")


print(f'RUNNING THE CODE FOR {number_of_iterations} ITERATIONS!')


if paths_per_request_cap:
    print(f'ASSERTING A MAXIMUM OF {max_num_paths_per_r} PATHS PER REQUEST!')
else:
    print(f'NOT ASSERTING A MAXIMUM PATHS PER REQUEST!')

if diversification:
    print(f'USING DIVERSIFICATION IN SELECTING NEW COLUMNS TO BE ADDED!')
    print(f'MEHTHOD: {div_method}')
    #print(f'A PATH WILL NOT BE ADDED IF, FOR ALL ITS PACKING COMPONENTS, A PATH ALREADY EXISTS FOR THE CURRENT ITERATION AND ULD.')
else:
    print(f'NOT USING DIVERSIFICATION!')

#%%
cwd = os.getcwd()

schedule_small = inputs.schedule_small
clustering_requests = inputs.clustering_requests
clustering_maximum = inputs.clustering_maximum
subset = inputs.subset
max_num_of_ulds_per_R = inputs.max_num_of_ulds_per_R
comp_level = inputs.comp_level
Payload = inputs.Payload
free_aircraft_placement = inputs.free_aircraft_placement
r_seed = inputs.r_seed

random.seed(r_seed)


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
A_all_dict_2 = setsparameters.A_all_dict_2
A_all_dict_3 = setsparameters.A_all_dict_3
K_f = setsparameters.K_f
G_u = setsparameters.G_u
F_u = setsparameters.F_u
U_v_f = setsparameters.U_v_f
U_f = setsparameters.U_f
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
identical_ulds = setsparameters.identical_ulds

R_dict = setsparameters.R_dict
R_avg = 4000  # Euros/ton
INC_dict_u = setsparameters.INC_dict_u


if schedule_small:
    folder_name = 'reduced'
if not schedule_small:
    folder_name = 'full'
folder_name = folder_name + '_' + comp_level
folder_name = folder_name + '_'+ str(Payload)
if free_aircraft_placement:
    folder_name = folder_name + '_free'
if not free_aircraft_placement:
    folder_name = folder_name + '_restricted'

if subset:
    folder_name = folder_name + '_subset' + str(max_num_of_ulds_per_R)
if clustering_requests:
    if clustering_maximum == 0:
        folder_name = folder_name + '_clusteringSmall'
    if clustering_maximum == 1:
        folder_name = folder_name + '_clusteringLarge'

folder_name = folder_name + '_'+str(number_of_iterations)+'it'

if linearize_PP:
    folder_name = folder_name + '_linPP'

if w_min:
    folder_name = folder_name + '_w_min'
if w_max:
    folder_name = folder_name + '_w_max'

if paths_per_request_cap:
    folder_name = folder_name + '_cap'+ str(max_num_paths_per_r)
if diversification:
    folder_name = folder_name + '_div' + div_method

folder_name = folder_name + '_seed' + str(setsparameters.r_seed)

print('FOLDER NAME:', folder_name)

time.sleep(5)
startTime_global = time.time()
#%%
# Define some lists and dicts to monitor the performance of the algorithm

copy_time_list = []
model_calc_time_list = []

paths_added_idx = {}

r_p_counter = {}
f_p_counter = {}
for r in R_dict.keys():
    r_p_counter[r] = 0
for f in F_dict.keys():
    f_p_counter[f] = 0
# %%
# Determine shortest paths between Origins and Destinations

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
SP_node = {}
SP_arc = {}
for o, n_o in source_node_airports.items():
    for d, n_d in target_node_airports.items():
        if d != o:
            try:
                path = nx.shortest_path(G, source=n_o, target=n_d)
                arcs = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                SP_node[(o, d)] = path
                SP_arc[(o, d)] = [next(k for k, v in A_all_dict_2.items() if v == arc) for arc in arcs]
            except (nx.NetworkXNoPath, KeyError):
                SP_node[(o, d)] = [-1]
                SP_arc[(o, d)] = [-1]
print(SP_arc)
print('Shortest paths calculated')
# %%
########################################################################################################################
############################################# SUPPORTING FUNCTIONS #####################################################
########################################################################################################################
def fits_in_ULD(packing_entries, request, u, arc_path, source_arc, sink_arc):
    ULD_weight = V_info[U_dict[u][2]][0]
    ULD_volume = V_info[U_dict[u][2]][1]

    ULD_dd = N_dict_idx_at[TSN_ULD_sink_dict[sink_arc][0]][1]
    ULD_rt = N_dict_idx_at[ULD_source_TSN_dict[source_arc][1]][1]

    #print('ULD_weight:', ULD_weight, 'ULD_volume:',ULD_volume)
    current_weight = sum(R_dict[packing][5] for packing in packing_entries)
    current_volume = sum(R_dict[packing][6] for packing in packing_entries)
    #print('current_weight:', current_weight, 'current_volume:',current_volume)
    #print(R[request][5], R[request][6])
    for inc in INC_dict_u[u].keys():
        for r in INC_dict_u[u][inc]:
            for packing in packing_entries:
                if (packing, request) == r or (request, packing) == r:
                    #print('Incompatible packing:', inc, packing, request)
                    return False
    if request in packing_entries:
        #print(f'Request {request} is already in ULD {u}')
        return False

    for f in arc_path:
        if f in F_dict.keys():
            if dict_IATA_idx[N_dict_idx_at[F_dict[f][0]][0]] == dict_IATA_idx[R_dict[request][0]]:
                if F_r_plus.get((request, u), False) or F_r_plus.get((request, u), False) == []:
                    if f not in F_r_plus[(request, u)]:
                        #print(f'Request {request} cannot be placed in ULD {u} as it cannot make use of the arc path '
                        #      f'{arc_path} as the request is only released at node '
                        #      f'{N_dict_at_idx[(R_dict[request][0],R_dict[request][10])]}:'
                        #      f'{(R_dict[request][0],R_dict[request][10])}')
                        return False
            if dict_IATA_idx[N_dict_idx_at[F_dict[f][1]][0]] == dict_IATA_idx[R_dict[request][1]]:
                if F_r_minus.get((request, u), False) or F_r_minus.get((request, u), False) == []:
                    if f not in F_r_minus[(request, u)]:
                        #print(f'Request {request} cannot be placed in ULD {u} as it cannot make use of the arc path '
                        #      f'{arc_path} as the request has its due date at node '
                        #      f'{N_dict_at_idx[(R_dict[request][1],R_dict[request][11])]}:'
                        #      f'{(R_dict[request][1],R_dict[request][11])}')
                        return False

    if R_dict[request][10] >= U_dict[u][0][1][1]:
        # print(f'Request {request} cannot be placed in ULD {u} as the release time of the request is after the due date '
        #    f'of the ULD.')
        return False
    if R_dict[request][11] <= U_dict[u][0][0][1]:
        # print(f'Request {request} cannot be placed in ULD {u} as the due date of the request is before the release time '
        #    f'of the ULD.')
        return False
    if request not in R_u[u]:
        # print(f'Request {request} cannot be placed in ULD {u} as it is not in U_r')
        return False

    if R_dict[request][5] + current_weight <= ULD_weight and R_dict[request][6] + current_volume <= ULD_volume:
        #print('it fits!')
        return True

def ULD_is_full(packing_entries, u):
    ULD_weight = V_info[U_dict[u][2]][0]
    ULD_volume = V_info[U_dict[u][2]][1]
    # print('ULD_weight:', ULD_weight, 'ULD_volume:',ULD_volume)
    current_weight = sum(R_dict[packing][5] for packing in packing_entries)
    current_volume = sum(R_dict[packing][6] for packing in packing_entries)

    smallest_weight = min((R_dict[index][5]) for index in R_u[u])
    smallest_volume = min((R_dict[index][6]) for index in R_u[u])

    if current_weight + smallest_weight >= ULD_weight or current_volume + smallest_volume >= ULD_volume:
        # print(f'ULD is full, break current_weight: {current_weight}, current_volume: {current_volume}, ULD_weight: {ULD_weight}, ULD_volume: {ULD_volume}')
        # print(f'smallest_weight: {smallest_weight}, smallest_volume: {smallest_volume}')
        return True

def add_unique_path_fast(new_entry, n_paths, unique_paths_set):
    new_entry_tuple = tuple((k, tuple(v) if isinstance(v, list) else v) for k, v in new_entry.items())
    if new_entry_tuple in unique_paths_set:
        #print("Path already exists, not adding.")
        return
    P_dict[n_paths] = new_entry
    unique_paths_set.add(new_entry_tuple)

def add_unique_path(new_entry, n_paths):
    for entry in P_dict.values():
        if entry == new_entry:
            #print("Path already exists, not adding.")
            return
    P_dict[n_paths] = new_entry
    # print("New path added successfully.")

def print_constraints_with_values(constraints, model):
    for constr in constraints:
        if constr.ConstrName.startswith('C7'):
            constr_expr = model.getRow(constr)
            expr_str = ""
            for i in range(constr_expr.size()):
                var = constr_expr.getVar(i)
                coeff = constr_expr.getCoeff(i)
                var_value = var.X
                expr_str += f"{coeff} * {var.VarName}({var_value}) + "
            expr_str = expr_str[:-3]  # Remove the last ' + '
            print(f"{constr.ConstrName} : {expr_str} <= {constr.RHS}")

# %%
########################################################################################################################
############################################### INITIAL PATH SETUP #####################################################
########################################################################################################################

class initial_path_setup:
    def __init__(self):
        self.P_dict = {}
        self.n_paths = 0
        self.W_up = {}
        self.R_up = {}
        self.C_up = {}
        self.unique_paths_set = set()
        self.P_u = {}
        self.P_uf = {}
        self.P_ur = {}
        self.P_arc = {}
        self.P_r = {}

    def run(self):

        for u, u_values in U_dict.items():
            ULD_source_TSN_arc = [u + len(A_all_dict_2)]
            TSN_ULD_sink_arc = [u + len(A_all_dict_2) + len(ULD_source_TSN_dict)]

            try:
                node_path = nx.shortest_path(G, source=u_values[1][0], target=u_values[1][1])
            except nx.NetworkXNoPath:
                print(f"No path found between {u_values[1][0]} and {u_values[1][1]}.")
                continue

            arcs = [(node_path[i], node_path[i + 1]) for i in range(len(node_path) - 1)]
            arc_path = [next(k for k, v in A_all_dict_2.items() if v == arc) for arc in arcs]
            path = arc_path + ULD_source_TSN_arc + TSN_ULD_sink_arc

            if -1 not in path:  # A possible arc path exists for this ULD
                self.P_dict[self.n_paths] = {"arc_path": sorted(path), "packing": []}
                if fits_in_ULD(self.P_dict[self.n_paths]["packing"], u_values[3], u, path, ULD_source_TSN_arc[0],
                               TSN_ULD_sink_arc[0]):
                    self.P_dict[self.n_paths]["packing"].append(u_values[3])
                sorted_R_u = random.sample(R_u[u], len(R_u[u]))
                for r in sorted_R_u:
                    if fits_in_ULD(self.P_dict[self.n_paths]["packing"], r, u, path, ULD_source_TSN_arc[0], TSN_ULD_sink_arc[0]):
                        self.P_dict[self.n_paths]["packing"].append(r)
                        if ULD_is_full(self.P_dict[self.n_paths]["packing"], u):
                            # print('ULD is full, break')
                            break
                self.P_dict[self.n_paths]["packing"].sort()
                print(u, self.n_paths, self.P_dict[self.n_paths])
                new_entry_tuple = tuple((k, tuple(v) if isinstance(v, list) else v) for k, v in self.P_dict[self.n_paths].items())
                self.unique_paths_set.add(new_entry_tuple)

            self.n_paths = len(self.P_dict)
        print(f'{len(self.P_dict)} paths were initialized for {len(U_dict.keys())} ULDs')

        # Define the cumulative weight and the cumulative revenue factor of the cargo requests carried by the considered ULD along path p
        for p, p_values in self.P_dict.items():
            requests = p_values["packing"]
            sum_weight = 0
            sum_rev = 0
            for r, r_values in R_dict.items():
                if r in requests:
                    sum_weight += r_values[5]
                    sum_rev += r_values[7] * r_values[5] * R_avg
            self.W_up[p] = sum_weight
            self.R_up[p] = sum_rev
        print('W_up & R_up set up')

        # Define the overall operational cost due to the transportation of ULD u along path p
        for p, p_values in self.P_dict.items():
            arcs = p_values["arc_path"]
            sum_dist = 0
            for f in F_dict.keys():
                if f in arcs:
                    sum_dist += F_info[f]['distance']
            C_ULD = V_info[U_dict[p][2]][5]
            self.C_up[p] = sum_dist * C_ULD
        print('C_up set up')

        for p, p_values in self.P_dict.items():
            for arc in p_values['arc_path']:
                if arc not in self.P_arc:
                    self.P_arc[arc] = []
                self.P_arc[arc].append(p)
            for packing in p_values['packing']:
                if packing not in self.P_r:
                    self.P_r[packing] = []
                self.P_r[packing].append(p)
        print('Index mappings set up')
        # Calculate P_u
        self.P_u = {u: self.P_arc.get(u + len(A_all_dict_2), []) for u in U_dict}
        print('P_u set up')
        # Calculate P_uf
        self.P_uf = {(u, f): list(set(self.P_u[u]).intersection(self.P_arc.get(f, []))) for u in U_dict for f in F_u[u]}
        print('P_uf set up')
        # Calculate P_ur
        self.P_ur = {(u, r): list(set(self.P_u[u]).intersection(self.P_r.get(r, []))) for u in U_dict for r in R_u[u]}
        print('P_ur set up')


# %%
########################################################################################################################
##################################### DEFINING THE RESTRICTED MASTER PROBLEM ###########################################
########################################################################################################################
class restricted_master_problem:
    def __init__(self, W_up, C_up, R_up, P_u, P_uf, P_ur):
        self.W_up = W_up
        self.C_up = C_up
        self.R_up = R_up
        self.P_u = P_u
        self.P_uf = P_uf
        self.P_ur = P_ur

    def build(self):
        self.master = Model()

        # Decision variables
        self.x = {}  # aircraft fleet routing decision variables
        self.z = {}  # ULD to path assignment decision variables

        # Aircraft routing decision variables for ground arcs (integer)
        for g in G_dict.keys():
            for k, v in K.items():
                self.x[g, k] = self.master.addVar(lb=0, ub=v['#'], vtype=GRB.INTEGER, name='x[%s,%s]' % (g, k))
        # Aircraft routing decision variables for flight arcs (binary)
        for f in F_dict.keys():
            for k in K_f[f]:
                self.x[f, k] = self.master.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='x[%s,%s]' % (f, k))
        # Aircraft routing decision variables from dummy source to TSN (integer)
        for a in Source_TSN_dict.keys():
            for k, v in K.items():
                self.x[a, k] = self.master.addVar(lb=0, ub=v['#'], vtype=GRB.INTEGER, name='x[%s,%s]' % (a, k))
        # Aircraft routing decision variables from TSN to dummy sink (integer)
        for a in TSN_Sink_dict.keys():
            for k, v in K.items():
                self.x[a, k] = self.master.addVar(lb=0, ub=v['#'], vtype=GRB.INTEGER, name='x[%s,%s]' % (a, k))
        # Aircraft routing decision variable from dummy source to dummy sink (integer)
        for k, v in K.items():
            for a in Source_Sink_dict.keys():
                self.x[a, k] = self.master.addVar(lb=0, ub=v['#'], vtype=GRB.INTEGER, name='x[%s,%s]' % (a, k))
        for u in U_dict.keys():
            for p in self.P_u[u]:
                self.z[u, p] = self.master.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='z[%s,%s]' % (u, p))

    # %%
    # CONSTRAINTS

        # Every flight arc can be flown at most by one aircraft type
        # This should be an equality constraint according to the manuscript meaning each flight would need to have an
        # aircraft type assigned, with <= some flight arcs can be left 'unflown'
        if free_aircraft_placement:
            self.C1 = self.master.addConstrs((quicksum(self.x[f, k] for k in K_f[f]) <= 1 for f in F_dict.keys()),
                               name="C1")
        if not free_aircraft_placement:
            self.C1 = self.master.addConstrs((quicksum(self.x[f, k] for k in K_f[f]) == 1 for f in F_dict.keys()),
                                             name="C1")
        # The overall number of aircraft of a specific type must leave the dummy source
        self.C2 = self.master.addConstrs(((quicksum(self.x[a, k] for a in Source_TSN_dict.keys()) + \
                                 quicksum(self.x[a, k] for a in Source_Sink_dict.keys()) ==
                                 N_k[k] for k in K.keys())), name="C2")
        # The overall number of aircraft of a specific type must arrive to the dummy sink
        self.C3 = self.master.addConstrs(((quicksum(self.x[a, k] for a in TSN_Sink_dict.keys()) + \
                                 quicksum(self.x[a, k] for a in Source_Sink_dict.keys()) ==
                                 N_k[k] for k in K.keys())), name="C3")
        # Conservation of flow in the TSN per aircraft type
        self.C4 = self.master.addConstrs((quicksum(self.x[a, k] for a in AK_k_n_plus[(k, n)]) -
                                quicksum(self.x[a, k] for a in AK_k_n_minus[(k, n)]) == 0
                                for k in K.keys() for n in N_K_realTSN.keys()), name="C4")
        # For each flight leg, we can carry as many ULDs of type v as there are allowed by the configuration
        self.C5 = self.master.addConstrs((quicksum(self.z[u, p] for u in U_v_f[(ULD_type_idx, f)] for p in self.P_uf[(u, f)]) -
                                quicksum(N_kv[k][ULD_type_idx] * self.x[f, k] for k in K_f[f]) <= 0
                                for f in F_dict.keys() for ULD_type_idx in U_uld_type_idx.keys()),
                               name="C5")
        # For each flight arc, check if payload capacity is not exceeded
        self.C6 = self.master.addConstrs((quicksum(self.W_up[p] * self.z[u, p] for u in U_f[f] for p in self.P_uf[(u, f)]) -
                                quicksum(W_fk[(f, k)] * self.x[f, k] for k in K_f[f]) <= 0
                                for f in F_dict.keys()),
                               name="C6")
        ## For each ULD at most one path can be selected
        self.C7 = self.master.addConstrs((quicksum(self.z[u, p] for p in self.P_u[u]) <= 1
                                for u in U_dict.keys()),
                               name="C7")
        # For each request at most one path containing it can be selected
        self.C8 = self.master.addConstrs((quicksum(self.z[u, p] for u in U_r[r] for p in self.P_ur[(u, r)]) <= 1
                                for r in R_dict.keys()),
                               name="C8")
        obj = LinExpr()
        # Revenue due to transporting requests
        for u in U_dict.keys():
            for p in self.P_u[u]:
                obj += self.R_up[p] * self.z[u, p]
        # Operational cost due to routing of aircraft fleet
        for f in F_dict.keys():
            for k in K_f[f]:
                obj -= K[k]['OC'] * F_info[f]['distance'] * self.x[f, k]
        # Operational cost due to routing of ULDs
        for u in U_dict.keys():
            for p in self.P_u[u]:
                obj -= self.C_up[p] * self.z[u, p]
        # Add a small cost to enter the TSN from the dummy source
        # to avoid symmetries
        eps = 0.01
        for a in Source_TSN_dict.keys():
            for k, v in K.items():
                obj -= eps * self.x[a, k]
        self.master.setObjective(obj, GRB.MAXIMIZE)
        self.master.update()
        #self.master.write('rmp_main.lp')

        return self.master, self.x, self.z


# %%
########################################################################################################################
############################################### PRICING PROBLEM ########################################################
########################################################################################################################
class pricing_problem:
    def __init__(self, u, count, gamma_fv, delta_f, eta_u, lambda_r, R_u=R_u):
        self.u = u
        self.count = count
        #self.R_u = copy.deepcopy(R_u)
        self.R_u = R_u[self.u].copy()
        self.gamma_fv = gamma_fv
        self.delta_f = delta_f
        self.eta_u = eta_u
        self.lambda_r = lambda_r
        self.num_new_paths_u = 0
        self.idx_new_paths_u = []
        self.unique_packings = []
        self.C_ULD = V_info[U_dict[self.u][2]][5]

        #self.paths_added_idx = paths_added_idx

    def execute(self):
        startTime_pp = time.time()

        self.preprocess()

        #if not self.R_u.get(self.u):
        #    print('Current u:', self.u, 'Current count', self.count, 'G_u:', len(G_u[self.u]), 'F_u:', len(F_u[self.u]),
        #          'R_u:', len(self.R_u[self.u]), 'No requests available for this ULD; skipping to the next one.')
        #    return self.num_new_paths_u, self.idx_new_paths_u
        if not self.R_u:
           print('Current u:', self.u, 'Current count', self.count, 'G_u:', len(G_u[self.u]), 'F_u:', len(F_u[self.u]),
                 'R_u:', len(self.R_u), 'No requests available for this ULD; skipping to the next one.')
           return self.num_new_paths_u, self.idx_new_paths_u

        PP = Model("Pricing Problem")
        y_pp, z_pp, w_pp = self.initialize_variables(PP)

        PP.update()
        variables = PP.getVars()
        # Get the number of variables
        num_vars = len(variables)
        print('Current u:', self.u, 'Current count', self.count, 'G_u:', len(G_u[self.u]), 'F_u:', len(F_u[self.u]), 'R_u:', len(self.R_u),
              'num_vars:', num_vars)
        self.setup_constraints(PP, y_pp, z_pp, w_pp)
        obj_PP = self.construct_objective(y_pp, z_pp, w_pp)

        PP.setObjective(obj_PP, GRB.MAXIMIZE)
        PP.setParam('OutputFlag', 0)
        # Enable solution pool
        PP.Params.PoolSearchMode = 2
        # Set the number of solutions to store
        PP.Params.PoolSolutions = number_of_solutions #max(20, 0.25 * len(self.R_u[self.u])) if self.R_u[self.u] else 1
        if diversification:
            PP.Params.PoolSolutions = number_of_solutions_div
        PP.optimize()
        # Iterate through each solution in the pool
        num_solutions = PP.SolCount
        self.process_solution(PP, num_solutions, y_pp, z_pp)

        PP.dispose()
        del PP
        del z_pp
        del y_pp
        del w_pp
        del variables

        print(f'Added {self.num_new_paths_u} new paths for ULD {self.u}')
        endTime_pp = time.time()
        # Calculate the elapsed time
        elapsed_time_calc = endTime_pp - startTime_pp
        # Print the elapsed time
        print("Elapsed time for this u:", elapsed_time_calc, "seconds")
        model_calc_time_list.append(elapsed_time_calc)

        return self.num_new_paths_u, self.idx_new_paths_u

    def preprocess(self):
        if SP_arc[(U_dict[self.u][0][0][0], U_dict[self.u][0][1][0])] == [-1]:
            print('No path possible for u:', self.u, 'skipping to next')
            return None

        if U_dict[self.u][2] in (0, 1):
            self.uld_type = 0
        if U_dict[self.u][2] in (2, 3):
            self.uld_type = 1

        if paths_per_request_cap:
            #for packing in self.R_u[self.u][:]:  # Iterate over a copy of the list
            #    if r_p_counter[packing] > max_num_paths_per_r:
            #        self.R_u[self.u].remove(packing)
            for packing in self.R_u[:]:  # Iterate over a copy of the list
               if r_p_counter[packing] > max_num_paths_per_r:
                   self.R_u.remove(packing)

    def initialize_variables(self, PP):
        y_pp = {}
        for g in G_u[self.u]:
            y_pp[g] = PP.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='y_pp[%s]' % (g))
        for f in F_u[self.u]:
            y_pp[f] = PP.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='y_pp[%s]' % (f))
        s = self.u + len(A_all_dict_2)
        y_pp[s] = PP.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='y_pp[%s]' % (s))
        t = self.u + len(A_all_dict_2) + len(ULD_source_TSN_dict)
        y_pp[t] = PP.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='y_pp[%s]' % (t))
        z_pp = {}
        #for r in self.R_u[self.u]:
        for r in self.R_u:
            z_pp[r] = PP.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='z_pp[%s]' % (r))

        w_pp = {}
        if linearize_PP:
            for f in F_u[self.u]:
                w_pp[f] = PP.addVar(lb=0, ub= 5, vtype=GRB.CONTINUOUS, name='w_pp[%s]' % (f))

        return y_pp, z_pp, w_pp

    def setup_constraints(self, PP, y_pp, z_pp, w_pp):
        # For each ULD, we must satisfy the weight capacity
        #PP_C1 = PP.addConstr((quicksum(R_dict[r][5] * z_pp[r] for r in self.R_u[self.u]) <= V_info[U_dict[self.u][2]][0]),
        #                     name="PP_C1")
        PP_C1 = PP.addConstr((quicksum(R_dict[r][5] * z_pp[r] for r in self.R_u) <= V_info[U_dict[self.u][2]][0]),
                            name="PP_C1")
        # For each ULD, we must satisfy the volume capacity
        #PP_C2 = PP.addConstr((quicksum(R_dict[r][6] * z_pp[r] for r in self.R_u[self.u]) <=
        #                      V_info[U_dict[self.u][2]][1]),
        #                     name="PP_C2")
        PP_C2 = PP.addConstr((quicksum(R_dict[r][6] * z_pp[r] for r in self.R_u) <=
                              V_info[U_dict[self.u][2]][1]),
                             name="PP_C2")
        # Conservation of flow in the of the path of each ULD
        PP_C3 = PP.addConstrs((quicksum(y_pp[a] for a in AU_u_n_plus[(self.u, n)]) -
                               quicksum(y_pp[a] for a in AU_u_n_minus[(self.u, n)]) == 0
                               for n in N_K_realTSN.keys()), name="PP_C3")
        #PP_C4 = PP.addConstrs((z_pp[r] -
        #                      quicksum(y_pp[f] for f in F_r_plus[r, self.u]) <= 0
        #                      for r in self.R_u[self.u]), name="PP_C4")
        PP_C4 = PP.addConstrs((z_pp[r] -
                               quicksum(y_pp[f] for f in F_r_plus[r, self.u]) <= 0
                               for r in self.R_u), name="PP_C4")
        #PP_C5 = PP.addConstrs((z_pp[r] -
        #                       quicksum(y_pp[f] for f in F_r_minus[r, self.u]) <= 0
        #                       for r in self.R_u[self.u]), name="PP_C5")
        PP_C5 = PP.addConstrs((z_pp[r] -
                               quicksum(y_pp[f] for f in F_r_minus[r, self.u]) <= 0
                               for r in self.R_u), name="PP_C5")
        #if INC_dict_u[self.u] and comp_level != "H":
        #    PP_C6 = PP.addConstrs((z_pp[r[0]] + z_pp[r[1]] <= 1)
        #                          for inc in INC_dict_u[self.u].keys() for r in INC_dict_u[self.u][inc] if r[0] in self.R_u[self.u] and r[1] in self.R_u[self.u])
        if INC_dict_u[self.u] and comp_level != "H":
           PP_C6 = PP.addConstrs((z_pp[r[0]] + z_pp[r[1]] <= 1)
                                 for inc in INC_dict_u[self.u].keys() for r in INC_dict_u[self.u][inc] if r[0] in self.R_u and r[1] in self.R_u)

        if linearize_PP:
            # Liniearization constraints
            M = 5  # Big M

            #PP_lin1 = PP.addConstrs((w_pp[f] >= quicksum(R_dict[r][5] * z_pp[r] for r in self.R_u[self.u]) - M*(1 - y_pp[f]) for f in F_u[self.u]), name='PP_lin1')
            #PP_lin2 = PP.addConstrs((w_pp[f] <= quicksum(R_dict[r][5] * z_pp[r] for r in self.R_u[self.u]) for f in F_u[self.u]), name='PP_lin2')
            PP_lin1 = PP.addConstrs((w_pp[f] >= quicksum(R_dict[r][5] * z_pp[r] for r in self.R_u) - M*(1 - y_pp[f]) for f in F_u[self.u]), name='PP_lin1')
            PP_lin2 = PP.addConstrs((w_pp[f] <= quicksum(R_dict[r][5] * z_pp[r] for r in self.R_u) for f in F_u[self.u]), name='PP_lin2')
            PP_lin3 = PP.addConstrs((w_pp[f] <= M * y_pp[f] for f in F_u[self.u]), name='PP_lin3')


    def construct_objective(self, y_pp, z_pp, w_pp):
        obj_PP = LinExpr()
        #C_ULD = 0.1

        if w_max:
            ###W_up = W_max:
            W_up_placeholder = V_info[U_dict[self.u][2]][0]

        if w_min:
            ###W_up = W_min:
            #min_R_u = min(self.R_u[self.u], key=lambda request: R_dict[request][5])
            min_R_u = min(self.R_u, key=lambda request: R_dict[request][5])
            W_up_placeholder = R_dict[min_R_u][5]

        if linearize_PP:
            W_up_placeholder = 0

        #for r in self.R_u[self.u]:
        for r in self.R_u:
            obj_PP += (R_avg * R_dict[r][7] * R_dict[r][5] - self.lambda_r[r]) * z_pp[r]
        for f in F_u[self.u]:
            obj_PP -= ((self.C_ULD * F_info[f]['distance'] + self.gamma_fv[(f, self.uld_type)] +
                        W_up_placeholder * self.delta_f[f]) * y_pp[f])
            # U_dict[u][2] is v, i.e. ULD type considering 4 types

        if linearize_PP:
            for f in F_u[self.u]:
                obj_PP -= (self.delta_f[f] * w_pp[f])

        # Constant in the obj function:
        obj_PP -= self.eta_u[self.u]
        return obj_PP

    def process_solution(self, PP, num_solutions, y_pp, z_pp):

        for sol in range(num_solutions):
            PP.setParam('SolutionNumber', sol)  # Set which solution to query
            # Access variables, objective value, etc., of the i-th solution
            # print("Solution", sol)
            # Access and print objective value
            # print("Objective value:", PP.PoolObjVal)
            if PP.PoolObjVal > 0:
                new_path_arcs = [i for i in y_pp if y_pp[i].Xn > 0.001]
                new_path_packing = [i for i in z_pp if z_pp[i].Xn > 0.001]

                if self.u == 0:
                    print(f'Obj value for iteration {self.count}, u {self.u}, solution {sol} = {PP.PoolObjVal}')

                if diversification:
                    if div_method == 'all':
                        if all(packing in self.unique_packings for packing in new_path_packing):
                            # print(f'For diverisfication the path {new_path_packing} is not considered since a solution covering all  '
                            #     f' constraints are already covered for this u {u}, {unique_packings_count_u}')
                            continue
                    if div_method == 'any':
                        if any(packing in self.unique_packings for packing in new_path_packing):
                            # print(f'For diverisfication the path {new_path_packing} is not considered since a solution covering all  '
                            #     f' constraints are already covered for this u {u}, {unique_packings_count_u}')
                            continue

                n_paths = len(P_dict)
                new_entry = {"arc_path": new_path_arcs, "packing": new_path_packing}
                add_unique_path_fast(new_entry, n_paths, unique_paths_set)

                if n_paths != len(P_dict):  # if a new path is added
                    self.idx_new_paths_u.append(n_paths)

                    for packing in new_path_packing:
                        r_p_counter[packing] += 1
                        if packing not in self.unique_packings:
                            self.unique_packings.append(packing)

                    for arc in new_path_arcs:
                        if arc in F_dict.keys():
                            f_p_counter[arc] += 1

                    sum_dist = sum(F_info[f]['distance'] for f in F_dict.keys() if f in new_path_arcs)
                    C_up[n_paths] = sum_dist * self.C_ULD

                    sum_weight = sum(R_dict[r][5] for r in new_path_packing)
                    sum_rev = sum(R_dict[r][7] * R_dict[r][5] * R_avg for r in new_path_packing)

                    W_up[n_paths] = sum_weight
                    R_up[n_paths] = sum_rev

                    col = Column()
                    for f in F_dict.keys():
                        if f in new_path_arcs:
                            col.addTerms(1, C5[(f, self.uld_type)])
                            col.addTerms(W_up[n_paths], C6[f])

                    col.addTerms(1, C7[self.u])

                    for r in new_path_packing:
                        col.addTerms(1, C8[r])

                    # col_dict[count][u][sol] = col
                    z_var[self.u, n_paths] = master.addVar(obj=R_up[n_paths] - C_up[n_paths], lb=0, ub=1, vtype=GRB.BINARY,
                                                  name='z[%s,%s]' % (self.u, n_paths), column=col)

                    n_new_paths = len(P_dict) - n_paths

                    if n_new_paths != 1:
                        print(f'n_new_paths is not 1!!! rather it is: {n_new_paths}')
                    self.num_new_paths_u += n_new_paths
                    if self.u == 0:
                        print(f'Path added {new_entry}')

# %%
########################################################################################################################
################################################# PATH COPIER ##########################################################
########################################################################################################################

class path_copier:
    def __init__(self, u, main_u, idx_new_paths_main_u):
        self.u = u
        self.main_u = main_u
        self.num_new_paths_u = 0
        self.idx_new_paths_main_u = idx_new_paths_main_u
        self.C_ULD = V_info[U_dict[self.u][2]][5]



    def copier(self):
        startTime_pc = time.time()
        self.prepocess()

        for path in self.idx_new_paths_main_u:

            path_to_copy = P_dict[path].copy()
            path_to_copy['arc_path'] = [arc for arc in path_to_copy['arc_path']
                                        if arc not in ULD_source_TSN_dict.keys()
                                        and arc not in TSN_ULD_sink_dict.keys()]
            ULD_source_TSN_arc = self.u + len(A_all_dict_2)
            TSN_ULD_sink_arc = self.u + len(A_all_dict_2) + len(ULD_source_TSN_dict)

            path_to_copy['arc_path'].append(ULD_source_TSN_arc)
            path_to_copy['arc_path'].append(TSN_ULD_sink_arc)

            able_to_add = True
            for packing in path_to_copy['packing'][:]:  # Iterate over a copy of the list
                if r_p_counter[packing] > max_num_paths_per_r:
                    able_to_add = False

            n_paths = len(P_dict)
            if able_to_add:
                add_unique_path_fast(path_to_copy, n_paths, unique_paths_set)

            if n_paths != len(P_dict):  # if a new path is added
                #paths_added_idx[self.u].append(n_paths)

                for packing in path_to_copy['packing']:
                    r_p_counter[packing] += 1
                for arc in path_to_copy['arc_path']:
                    if arc in F_dict.keys():
                        f_p_counter[arc] += 1

                sum_dist = sum(F_info[f]['distance'] for f in F_dict.keys() if f in path_to_copy['arc_path'])
                C_up[n_paths] = sum_dist * self.C_ULD

                sum_weight = sum(R_dict[r][5] for r in path_to_copy['packing'])
                sum_rev = sum(R_dict[r][7] * R_dict[r][5] * R_avg for r in path_to_copy['packing'])

                W_up[n_paths] = sum_weight
                R_up[n_paths] = sum_rev

                col = Column()
                for f in F_dict.keys():
                    if f in path_to_copy['arc_path']:
                        col.addTerms(1, C5[(f, self.uld_type)])
                        col.addTerms(W_up[n_paths], C6[f])

                col.addTerms(1, C7[self.u])

                for r in path_to_copy['packing']:
                    col.addTerms(1, C8[r])

                # col_dict[count][u][sol] = col
                z_var[self.u, n_paths] = master.addVar(obj=R_up[n_paths] - C_up[n_paths], lb=0, ub=1, vtype=GRB.BINARY,
                                              name='z[%s,%s]' % (self.u, n_paths), column=col)

                n_new_paths = len(P_dict) - n_paths

                if n_new_paths != 1:
                    print(f'n_new_paths is not 1!!! rather it is: {n_new_paths}')
                self.num_new_paths_u += n_new_paths


            # print(path_to_copy)
        print(f'Added {self.num_new_paths_u} new paths for ULD {self.u}')
        endTime_pc = time.time()
        # Calculate the elapsed time
        elapsed_time_copy = endTime_pc - startTime_pc
        copy_time_list.append(elapsed_time_copy)
        # Print the elapsed time
        print("Elapsed time for this u:", elapsed_time_copy, "seconds")

        return self.num_new_paths_u

    def prepocess(self):
        if U_dict[self.u][2] in (0, 1):
            self.uld_type = 0
        if U_dict[self.u][2] in (2, 3):
            self.uld_type = 1

# %%
########################################################################################################################
########################################### COLUMN GENERATION ##########################################################
########################################################################################################################


class column_generation:
    def __init__(self):
        self.count = 0
        self.lin_relax_sol = {}
        self.new_paths_dict = {}

        self.RMP_sol_dict = {}
        self.paths_per_it_dict = {}

        self.exit_condition = False


    def perform(self):
        while self.exit_condition is False and self.count < number_of_iterations:
            self.lin_relax_sol[self.count] = []
            self.new_paths_iteration = 0
            self.idx_new_paths_u = {}

            lin_relax = master.relax()
            lin_relax.optimize()

            for var in lin_relax.getVars():
                if var.X != 0:
                    self.lin_relax_sol[self.count].append([var.varName, var.X])

            self.duals = {}
            self.slacks = {}

            for c in lin_relax.getConstrs():
                self.duals[c.ConstrName] = c.Pi
                self.slacks[c.ConstrName] = c.Slack
            self.gamma_fv = {}
            self.delta_f = {}
            self.eta_u = {}
            self.lambda_r = {}

            # Print dual variables
            for constraint_name, dual_value in self.duals.items():
                # if dual_value > 0:
                # print(f"Dual variable for constraint {constraint_name}: {dual_value}")
                # Assign dual values for constraint C5 corresponding to gamma_fv
                if constraint_name.startswith('C5'):
                    indices = constraint_name.split('[')[1][:-1]
                    indices = tuple(map(int, indices.split(',')))
                    self.gamma_fv[indices] = dual_value
                # Assign dual values for constraint C6 corresponding to delta_f
                if constraint_name.startswith('C6'):
                    indices = int(constraint_name.split('[')[1][:-1])
                    self.delta_f[indices] = dual_value
                ## Assign dual values for constraint C7 corresponding to eta_u
                if constraint_name.startswith('C7'):
                    indices = int(constraint_name.split('[')[1][:-1])
                    self.eta_u[indices] = dual_value
                # Assign dual values for constraint C8 corresponding to lambda_r
                if constraint_name.startswith('C8'):
                    indices = int(constraint_name.split('[')[1][:-1])
                    self.lambda_r[indices] = dual_value

            print(f'DELTA: {self.delta_f}')

            for u in U_dict.keys():
                main_u = next((key for key, values in identical_ulds.items() if u in values), None)

                main_u_list = [key for key, values in identical_ulds.items() if u in values]  # Sanity check
                if len(main_u_list) > 1:
                    print(f'WE HAVE A PROBLEM!!!')
                    print(f'ULD {u} is said to have more than one main uld: {main_u_list}')

                if bool(main_u):
                    print(f'Current u: {u}, Current count: {self.count}.'
                          f' PP already solved for identical ULD {main_u}, adding the same paths to P_dict for ULD {u}')
                    PC = path_copier(u, main_u, self.idx_new_paths_u[main_u])
                    num_new_paths = PC.copier()
                    del PC

                else:
                    PP_class = pricing_problem(u, self.count, self.gamma_fv, self.delta_f, self.eta_u, self.lambda_r)
                    num_new_paths, self.idx_new_paths_u[u] = PP_class.execute()
                    del PP_class


                self.new_paths_iteration += num_new_paths

                print(f'Added {num_new_paths} new paths for ULD {u}')


            master.update()
            #master.write('rmp_update_main.lp')
            print(f'Number of paths added in iteration {self.count}: {self.new_paths_iteration}')
            self.RMP_sol_dict[self.count] = lin_relax.objVal
            self.paths_per_it_dict[self.count] = self.new_paths_iteration
            if self.new_paths_iteration == 0:
                self.exit_condition = True
            self.count += 1

        return self.RMP_sol_dict, self.paths_per_it_dict


#%%
########################################################################################################################
################################################# MAIN CODE ############################################################
########################################################################################################################

initial_paths = initial_path_setup()
initial_paths.run()

P_dict, W_up, C_up, R_up, P_u, P_uf, P_ur, unique_paths_set = \
    (initial_paths.P_dict, initial_paths.W_up, initial_paths.C_up, initial_paths.R_up,
    initial_paths.P_u, initial_paths.P_uf, initial_paths.P_ur, initial_paths.unique_paths_set)


RMP = restricted_master_problem(W_up, C_up, R_up, P_u, P_uf, P_ur)
master, x_var, z_var = RMP.build()
C5, C6, C7, C8 = RMP.C5, RMP.C6, RMP.C7, RMP.C8

CG = column_generation()
RMP_sol_dict, paths_per_it_dict = CG.perform()


endTime_global = time.time()
# Calculate the elapsed time
elapsed_time_global = endTime_global - startTime_global
# Print the elapsed time
print("Total elapsed time:", elapsed_time_global, "seconds")
print(f'Number of times the pricing problem was executed {len(model_calc_time_list)} with average time duration of {sum(model_calc_time_list) / len(model_calc_time_list)}')
if len(copy_time_list) != 0:
    print(f'Number of times existing paths were copied {len(copy_time_list)} with average time duration of {sum(copy_time_list) / len(copy_time_list)}')
print(f'Solutions of the RMP per iteration: {RMP_sol_dict}')
print(f'Number of new paths added per iteration: {paths_per_it_dict}')
#%%
########################################################################################################################
############################################# SAVE AND STORE PATHS #####################################################
########################################################################################################################

P_u = {u: [p for p, p_values in P_dict.items() if u + len(A_all_dict_2) in p_values['arc_path']] for u in U_dict}
P_r = {r: [p for p, p_values in P_dict.items() if r in p_values['packing']] for r in R_dict.keys()}
P_f = {f: [p for p, p_values in P_dict.items() if f in p_values['arc_path']] for f in F_dict.keys()}

p_dict_file = "P_dict.pickle"


full_folder_path = os.path.join("pickle_files", folder_name+"_CG")
os.makedirs(full_folder_path, exist_ok=True)
p_dict_file = os.path.join(full_folder_path, p_dict_file)

# Open file in binary write mode and dump the dictionary
with open(p_dict_file, "wb") as file:
    pickle.dump(P_dict, file)
    pickle.dump(C_up, file)
    pickle.dump(R_up, file)
    pickle.dump(W_up, file)
    pickle.dump(paths_per_it_dict, file)
    pickle.dump(elapsed_time_global, file)

print("Path dictionary saved to", p_dict_file)
print(f'Number of paths in total: {len(P_dict)}')

log_path = os.path.join("logs", folder_name+"_CG.log")

#%%
########################################################################################################################
########################################## REINSTATE INTEGRALITY CONSTRAINTS ###########################################
########################################################################################################################

master.setParam('MIPGap', 0.005)
master.setParam('FeasibilityTol', 1e-9)
master.setParam('TimeLimit', (3600) - elapsed_time_global)  # seconds
master.setParam("LogFile", log_path)
master.optimize()

#%%
########################################################################################################################
############################################# SAVE AND STORE VARIABLES #################################################
########################################################################################################################
final = []
for var in master.getVars():
    if var.Xn > 0.01:
        final.append([var.varName, var.Xn])
print(final)

x_dict = {}
y_dict = {}
z_dict = {}

PATH_uld_routing_costs = 0
PATH_revenue = 0

# Populate the dictionaries with appropriate values
for item in final:
    if item[0].startswith('x'):
        # Extract the indices from the string
        indices_str = item[0][2:-1]  # Remove 'x[' at the start and ']' at the end
        index_tuple = tuple(map(int, indices_str.split(',')))  # Convert to tuple of integers
        x_dict[index_tuple] = round(item[1])
    elif item[0].startswith('z'):
        # Extract the indices from the string
        indices_str = item[0][2:-1]  # Remove 'z[' at the start and ']' at the end
        index_tuple = tuple(map(int, indices_str.split(',')))  # Convert to tuple of integers
        u_value = index_tuple[0]
        p_value = index_tuple[1]
        if p_value in P_dict:
            arc_path_tuple = P_dict[p_value]['arc_path']
            packing_tuple = P_dict[p_value]['packing']
            for a in arc_path_tuple:
                y_dict[(a, u_value)] = round(item[1])
            for r in packing_tuple:
                z_dict[(r, u_value)] = round(item[1])
            PATH_uld_routing_costs -= C_up[p_value]
            PATH_revenue += R_up[p_value]

# Specify the file names
x_dict_file = os.path.join(full_folder_path, 'x_data.pickle')
y_dict_file = os.path.join(full_folder_path, 'y_data.pickle')
z_dict_file = os.path.join(full_folder_path, 'z_data.pickle')

# Write the dictionaries to pickle files
with open(x_dict_file, 'wb') as file:
    pickle.dump(x_dict, file)

with open(y_dict_file, 'wb') as file:
    pickle.dump(y_dict, file)
    pickle.dump(PATH_uld_routing_costs, file)

with open(z_dict_file, 'wb') as file:
    pickle.dump(z_dict, file)
    pickle.dump(PATH_revenue, file)

print(f"Dictionaries of x, y, and z values have been written to {x_dict_file}, {y_dict_file}, and {z_dict_file} respectively.")
print(f"(Including path uld routing costs and path revenue in {y_dict_file}, and {z_dict_file} respectively.)")


# %%
########################################################################################################################
################################################### PLOTS ##############################################################
########################################################################################################################
# Initialize lists to store the percentages and users
percentages = []
ULDs = []
for u in U_dict.keys():
    # Requests in available paths
    paths_for_u = P_u[u]
    unique_packings_u = []
    for path_key in paths_for_u:
        # Get the packing entries for the current path
        packing_entries = P_dict[path_key]["packing"]
        # Add unique packing entries to the list
        for packing in packing_entries:
            if packing not in unique_packings_u:
                unique_packings_u.append(packing)
    # Total requests that can be taken by u
    # Calculate percentage
    percentage = (len(unique_packings_u) / len(R_u[u])) * 100 if len(R_u[u]) != 0 else 0
    #if percentage > 100:
    #    print(u)
    #    print(percentage)
    #    print(unique_packings, len(unique_packings))
    #    print(R_u[u], len(R_u[u]))
    # Append to lists
    percentages.append(percentage)
    ULDs.append(u)
num_paths_for_r = []
requests = []

for r in R_dict.keys():
    paths_for_r = P_r[r]
    num_paths_for_r.append(len(paths_for_r))
    requests.append(r)
num_paths_for_uf = []
ULDs2 = []
for u in U_dict.keys():
    # Requests in available paths
    paths_for_u = P_u[u]
    unique_paths = []
    for path_key in paths_for_u:
        # Get the packing entries for the current path
        arc_paths = [arc for arc in P_dict[path_key]["arc_path"] if arc in F_dict.keys()]
        # Add unique packing entries to the list
        if arc_paths not in unique_paths:
            unique_paths.append(arc_paths)
    num_paths_for_uf.append(len(unique_paths))
    ULDs2.append(u)
num_paths_for_f = []
f_arcs = []
for f in F_dict.keys():
    paths_for_f = P_f[f]
    num_paths_for_f.append(len(paths_for_f))
    f_arcs.append(f)
# Plot the percentages as a bar chart
plt.figure(figsize=(10, 6))
plt.bar(requests, num_paths_for_r, color='skyblue', width=1)
plt.xlabel('Requests')
plt.ylabel('Paths')
plt.title('Paths containing request r')
plt.xticks(rotation=45)
plt.tight_layout()
# Plot the percentages as a bar chart
plt.figure(figsize=(10, 6))
plt.bar(ULDs, percentages, color='skyblue', width=1)
plt.xlabel('ULDs')
plt.ylabel('Percentage (%)')
plt.title('Percentage of available requests per ULD that are in a path for said ULD')
plt.xticks(rotation=45)
plt.tight_layout()
# Plot the percentages as a bar chart
plt.figure(figsize=(10, 6))
plt.bar(ULDs2, num_paths_for_uf, color='skyblue', width=1)
plt.xlabel('ULDs')
plt.ylabel('Paths')
plt.title('Unique arc paths considered per ULD')
plt.xticks(rotation=45)
plt.tight_layout()
plt.figure(figsize=(10, 6))
plt.bar(f_arcs, num_paths_for_f, color='skyblue')
plt.xlabel('Flight Arc')
plt.ylabel('Paths')
plt.title('Number of paths using a flight arc')
plt.xticks(rotation=45)
plt.tight_layout()


#if __name__ == "__main__":
#    plt.show()