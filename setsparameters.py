# -*- coding: utf-8 -*-
"""
Created on Wed April 11 11:37:03 2024

@author: luuk barbian
"""

import pandas as pd
from datetime import datetime, timezone
import time
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from itertools import chain
import inputs


start_time = time.time()
cwd = os.getcwd()

#%%
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

# read by default 1st sheet of an excel file
if schedule_small:
    print('USING REDUCED SCHEDULE!')
    df = pd.read_excel('instance_generation/instance_data.xlsx', 'schedule_reduced')
if not schedule_small:
    print('USING FULL SCHEDULE!')
    df = pd.read_excel('instance_generation/instance_data.xlsx', 'schedule_full')

if clustering_requests:
    if clustering_maximum == 0:
        print('CLUSTERING REQUESTS WITH A MAXIMUM SIZE OF THE SMALL ULD PER CLUSTER!')
    if clustering_maximum == 1:
        print('CLUSTERING REQUESTS WITH A MAXIMUM SIZE OF THE LARGCE ULD PER CLUSTER!')
if not clustering_requests:
    print('NOT CLUSTERING REQUESTS')

if subset:
    print(f'USING A MAXIMUM NUMBER OF {max_num_of_ulds_per_R} ULDs PER REQUEST!')
if not subset:
    print('NO LIMITATION ON ULDs PER REQUEST')

if comp_level == 'L':
    print(f'LOW COMPATIBILITY LEVEL')

if comp_level == 'M':
    print(f'MEDIUM COMPATIBILITY LEVEL')

if comp_level == 'H':
    print(f'HIGH COMPATIBILITY LEVEL')

# Payload in tonnes per outgoing flight assumed
print(f'USING {Payload} TONNES OF PAYLOAD PER OUTGOING FLIGHT!!!')

if not free_aircraft_placement:
    print(f'AIRCRAFT ONLY START AND END AT HUB AND ALL FLIGHTS ARE MANDATORY')
if free_aircraft_placement:
    print(f'FREE AIRCRAFT PLACEMENT AND ALL FLIGHTS ARE OPTIONAL')

# %%
########################################################################################################################
###################################### INITIAL TSN NETWORK #############################################################
########################################################################################################################

nodes = []
origin = []
destination = []
for row in df.loc[:, ['origin_airport', 'departure_time']].itertuples():
    nodes.append((row[1], int(row[2].timestamp())))
    origin.append((row[1], int(row[2].timestamp())))

for row in df.loc[:, ['destination_airport', 'arrival_time']].itertuples():
    nodes.append((row[1], int(row[2].timestamp())))
    destination.append((row[1], int(row[2].timestamp())))

df_nodes = pd.DataFrame(nodes, columns=['airport', 'time'])
df_nodes.sort_values(by=['airport', 'time'], ascending=[True, True], inplace=True)

airports = df_nodes.airport.unique()
ground_arcs = []

for a in airports:
    filter_ = df_nodes['airport'] == a
    aux = df_nodes[filter_]
    if len(aux) > 1:
        ground_aux = [
            ((aux['airport'].iloc[i], aux['time'].iloc[i]), (aux['airport'].iloc[i + 1], aux['time'].iloc[i + 1])) for i
            in range(0, len(aux) - 1)]
        ground_arcs.extend(ground_aux)

flight_arcs = [(origin[i], destination[i]) for i in range(len(origin))]

# Original nodes
N = []
for f in flight_arcs:
    N.append(f[0])
    N.append(f[1])

# Adding source and sink nodes

T_min = 0
T_max = max([t[1] for t in N]) + 1

Source_node = ('Source', T_min)
Sink_node = ('Sink', T_max)
N.append(Source_node)
N.append(Sink_node)

N_dict_idx_at = {k: v for k, v in enumerate(N)}
N_dict_at_idx = {v: k for k, v in N_dict_idx_at.items()}

earliest_node_airport = {k: sorted([v for idx, v in N_dict_idx_at.items() if v[0] == k],
                                   key=lambda a: a[1])[0]
                         for k in airports}
latest_node_airport = {k: sorted([v for idx, v in N_dict_idx_at.items() if v[0] == k],
                                 key=lambda a: a[1])[-1]
                       for k in airports}
nodes_at_airport = {k: sorted([v for idx, v in N_dict_idx_at.items() if v[0] == k],
                                 key=lambda a: a[1])
                       for k in airports}
dummy_source_TSN = []
TSN_dummy_sink = []
if free_aircraft_placement:
    for a in airports:
        filter_ = df_nodes['airport'] == a
        aux = df_nodes[filter_]
        dummy_aux = [(Source_node, (aux['airport'].iloc[0], aux['time'].iloc[0]))]
        dummy_source_TSN.extend(dummy_aux)
        TSN_aux = [((aux['airport'].iloc[-1], aux['time'].iloc[-1]), Sink_node)]
        TSN_dummy_sink.extend(TSN_aux)

if not free_aircraft_placement:
    filter_ = df_nodes['airport'] == 'AMS'
    aux = df_nodes[filter_]
    dummy_aux = [(Source_node, (aux['airport'].iloc[0], aux['time'].iloc[0]))]
    dummy_source_TSN.extend(dummy_aux)
    TSN_aux = [((aux['airport'].iloc[-1], aux['time'].iloc[-1]), Sink_node)]
    TSN_dummy_sink.extend(TSN_aux)

G_dict_idx_at = {k: v for k, v in enumerate(ground_arcs)}
F_dict_idx_at = {k + len(G_dict_idx_at): v for k, v in enumerate(flight_arcs)}
G_dict_at_idx = {v: k for k, v in G_dict_idx_at.items()}
F_dict_at_idx = {v: k for k, v in F_dict_idx_at.items()}
Source_TSN_idx_at = {k + len(G_dict_idx_at) + len(F_dict_idx_at)
                     : v for k, v in enumerate(dummy_source_TSN)}
TSN_Sink_idx_at = {k + len(G_dict_idx_at) + len(F_dict_idx_at)
                   + len(Source_TSN_idx_at): v for k, v in enumerate(TSN_dummy_sink)}
########################################################################################################################
########################################################################################################################
########################################################################################################################

#%%
#################################### NETWORK TOOL THAT IS USED FOR PATH FINDING ########################################

G_dict          = {k:(N_dict_at_idx[v[0]],N_dict_at_idx[v[1]]) for k,v in G_dict_idx_at.items()}    # Set of all ground arcs
F_dict          = {k:(N_dict_at_idx[v[0]],N_dict_at_idx[v[1]]) for k,v in F_dict_idx_at.items()}    # Set of all flight arcs
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

########################################################################################################################

#%%
########################################################################################################################
############################################## DEMAND DATA SET UP ######################################################
########################################################################################################################

flight_o = {}
for a in airports:
    filter_ = df['origin_airport'] == a
    flight_o[a] = len(df[filter_]) * Payload

flight_d = {}
for a in airports:
    filter2_ = df['destination_airport'] == a
    flight_d[a] = len(df[filter2_]) * Payload
# Proportion of demand between each region. Must satisfy that sum_j p_ij=1 for each i.
# Airports and the region they belong to
# Proportions of importance for each airport for the regional demand

if schedule_small:
    df_a = pd.read_excel('instance_generation/instance_data.xlsx', sheet_name='airports_reduced')
    df_d = pd.read_excel('instance_generation/instance_data.xlsx', index_col=0, sheet_name='demand_reduced')

else:
    df_a = pd.read_excel('instance_generation/instance_data.xlsx', sheet_name='airports_full')
    df_d = pd.read_excel('instance_generation/instance_data.xlsx', index_col=0, sheet_name='demand_full')

# Compute the weight to be transported between each possible origin destination pair, based on historic cargo data.
weight = {}
for a in airports:
    tot_w = flight_o[a]  # Total weight originating in airport a
    #print(a, tot_w)
    filter_ = df_a['airport'] == a
    aux = df_a[filter_]
    region_aux = aux['region'].iloc[0]  # The region where airport a belongs to

    # Iterate over all airports in the specific region
    for a2 in airports:
        if a2 != a:
            weight[(a, a2)] = float("{:.2f}".format(df_d.at[a2, a] * tot_w))


print(weight)
# Data commodity types per specific intercontinental cargo flows
commodity_df = pd.read_excel('instance_generation/instance_data.xlsx', index_col=0, sheet_name='commodity')

# Characteristics per commodity type (think of moving to excell file too)
request_characteristics_commodity_type = \
    {'chemical': {'weight': [0.4, 0.8], 'density': [0.15, 0.25],
                  'str_fct': [0.7, 1.3], 'perc_spec_ft': {'L': [1, 0.6],
                                                          'M': [0.5, 0.6], 'H': [0.25, 0.25]}},
     'perishable': {'weight': [0.4, 0.8], 'density': [0.15, 0.20],
                    'str_fct': [1, 1.3], 'perc_spec_ft': {'L': [0.3, 0.6],
                                                          'M': [0.0, 0.5], 'H': [0, 0.25]}},
     'heavy': {'weight': [0.4, 0.8], 'density': [0.5, 0.7],
               'str_fct': [0.5, 0.8], 'perc_spec_ft': {'L': [0.4, 0.8],
                                                       'M': [0.2, 0.3], 'H': [0.1, 0.1]}},
     'other': {'weight': [0.4, 0.8], 'density': [0.2, 0.3],
               'str_fct': [0.9, 1.1], 'perc_spec_ft': {'L': [0.25, 0.25],
                                                       'M': [0, 0], 'H': [0, 0]}}, }

#%%
########################################################################################################################
############################################## DEMAND GENERATION #######################################################
########################################################################################################################

def pick_times(nodes_at_airport, origin, destination):
    # Get the list of times for the origin and destination airports
    origin_times = [node[1] for node in nodes_at_airport[origin]]
    destination_times = [node[1] for node in nodes_at_airport[destination]]

    # Shuffle the origin times to ensure randomness
    random.shuffle(origin_times)

    for release_time in origin_times:
        # Filter the destination times to get only times after the release time
        possible_due_times = []
        for time in destination_times:
            if time > release_time:
                try:
                    path = nx.shortest_path(G, source=N_dict_at_idx[origin, release_time],
                                            target=N_dict_at_idx[destination, time])
                    possible_due_times.append(time)
                except nx.NetworkXNoPath:
                    pass

        if possible_due_times:
            # Pick a random due time from the possible due times
            due_time = random.choice(possible_due_times)
            #print(f"Release time: {release_time}, Due time: {due_time}")  # Debugging print statement
            return release_time, due_time

    # If no valid combination found, raise an error
    raise ValueError(f"No valid release and due time combination found for {origin} and {destination}.")

R_dict = {}
count = 0

for od, rem_weight in weight.items():
    o = od[0]
    d = od[1]

    r_o = df_a[df_a.airport == o].region.values[0]
    r_d = df_a[df_a.airport == d].region.values[0]
    perc_comm = commodity_df[str(r_o)+' - '+str(r_d)]
    while rem_weight > 0:
        rnd = random.random()
        if rnd <= perc_comm.loc['chemical']:
            commodity_type = 'chemical'
        elif rnd <= perc_comm.loc['chemical'] + perc_comm.loc['perishable']:
            commodity_type = 'perishable'
        elif rnd <= perc_comm.loc['chemical'] + perc_comm.loc['perishable'] + perc_comm.loc['heavy']:
            commodity_type = 'heavy'
        else:
            commodity_type = 'other'

        t_rt, t_dd = pick_times(nodes_at_airport, o, d)

        weight_r = np.round(random.uniform(request_characteristics_commodity_type[commodity_type]['weight'][0],
                                           request_characteristics_commodity_type[commodity_type]['weight'][1]), 1)
        volume_r = np.round(
            weight_r / (random.uniform(request_characteristics_commodity_type[commodity_type]['density'][0],
                                       request_characteristics_commodity_type[commodity_type]['density'][1])), 1)
        str_fct_r = np.round(random.uniform(request_characteristics_commodity_type[commodity_type]['str_fct'][0],
                                            request_characteristics_commodity_type[commodity_type]['str_fct'][1]), 1)

        R_dict[count] = (o, d, r_o, r_d, commodity_type, weight_r, volume_r, str_fct_r,
                   1 if random.random() <=
                        request_characteristics_commodity_type[commodity_type]['perc_spec_ft'][comp_level][0] else 0,
                   1 if random.random() <=
                        request_characteristics_commodity_type[commodity_type]['perc_spec_ft'][comp_level][1] else 0,
                   t_rt, t_dd)
        # print(count, R_dict[count])
        # R_dict = [('ORIGIN AIRPORT', 'DESTINATION AIRPORT', 'ORIGIN REGION', 'DESTINATION REGION', 'COMODITY TYPE',
        # Weight, Volume, Strategic factor from average revenue, 1 if hazardous, 1 if temp. controlled, release time,
        # due time)]
        count += 1
        rem_weight -= weight_r

R_dict_group = {}
count = 0
assigned_requests = []

if clustering_maximum == 0:
    max_cluster_weight = 1.5
    max_cluster_volume = 4.5
if clustering_maximum == 1:
    max_cluster_weight = 4.6
    max_cluster_volume = 10.7


if clustering_requests:
    for r, r_values in R_dict.items():
        if r not in assigned_requests:

            new_values = list(r_values)  # Convert tuple to list to modify it
            new_values.append([r])  # Append the new column value (r) to the list
            assigned_requests.append(r)


            for r2, r2_values in R_dict.items():

                if r2 != r and r2 not in assigned_requests and new_values[0] == r2_values[0] and new_values[1] == r2_values[1]\
                and new_values[4] == r2_values[4] and new_values[5] + r2_values[5] <= max_cluster_weight and new_values[6] + r2_values[6] <= max_cluster_volume\
                and new_values[8] == r2_values[8] and new_values[9] == r2_values[9] and abs(new_values[10] - r2_values[10]) <= 10000\
                and abs(new_values[11] - r2_values[11]) <= 10000 and len(new_values[12]) + 1 <=3:
                    new_values[12].append(r2)
                    new_values[7] = (new_values[7] * new_values[5] + r2_values[7] * r2_values[5])/(new_values[5] + r2_values[5])

                    new_values[5] += r2_values[5]
                    new_values[6] += r2_values[6]

                    new_values[10] = max(new_values[10], r2_values[10])
                    new_values[11] = min(new_values[11], r2_values[11])


                    assigned_requests.append(r2)

            R_dict_group[count] = tuple(new_values)  # Convert back to tuple and store in R_dict_group
            count += 1

    R_dict = R_dict_group

    for r, r_values in R_dict_group.items():
        if r_values[5] > 4.6:
            print('WEIGHT TOO HIGH')
        if r_values[6] > 10.7:
            print('VOLUME TOO LARGE')
        print(r, r_values)


# Define the incompatiblity sets
inc_combo = [('perishable', 'heavy'), ('perishable', 'chemical')]
INC_dict = {k: [(r1, r2) for idx_r1, r1 in enumerate(R_dict.keys())
                for idx_r2, r2 in enumerate(R_dict.keys()) if (idx_r2 > idx_r1
                                                          and ((R_dict[r1][4] == k[0] and R_dict[r2][4] == k[1]) or
                                                               (R_dict[r1][4] == k[1] and R_dict[r2][4] == k[0])))] for k in
            inc_combo}

########################################################################################################################
########################################################################################################################
########################################################################################################################
#%%
########################################################################################################################
########################################### ULD GENERATION #############################################################
########################################################################################################################
# Definition of ULD types
n_spec_feat = 2

V_info = {0: [1.5, 4.5, 1, 0, 'LD3', 0.105],  # LD3,     weight [ton], volume [m^3], 1 if hazardous, 1 if temp. controlled
          1: [1.5, 4.5, 1, 1, 'LD3', 0.115],  # LD3plus, weight [ton], volume [m^3], 1 if hazardous, 1 if temp. controlled
          2: [4.6, 10.7, 0, 0, 'LD7', 0.1],  # LD7,     weight [ton], volume [m^3], 1 if hazardous, 1 if temp. controlled
          3: [4.6, 10.7, 1, 1, 'LD7', 0.135]}  # LD7plus, weight [ton], volume [m^3], 1 if hazardous, 1 if temp. controlled

V_info_simp =  {0: {'Weight': 1.5, 'Volume': 4.5, 'Name':'LD3'},  # LD3,   weight [ton], volume [m^3]
                1: {'Weight': 4.6, 'Volume': 10.7, 'Name':'LD7'}}  # LD7,   weight [ton], volume [m^3]

V_ULD_type = ['LD3', 'LD7']          # Set of ULD types: V
V_ULD_type_idx_dict = {ULD_type: [k for k, v in V_info.items() if v[4] == ULD_type] for ULD_type in V_ULD_type}

def comp_ULD_check(r, V_info, ULD_type, n_spec_feat):
    comp_features = [1 if V_info[ULD_type][2 + i] >= r[8 + i] else 0 for i in range(0, n_spec_feat)]
    return all([c == 1 for c in comp_features])
def comp_ULD_new(r, V_info, n_spec_feat):
    # Defining if the ULD is suitable for hazardous or temp. controlled cargo
    ULD_type_comp = []
    for v in range(0, len(V_info)):
        this_ULD_type = []
        #if r[8] == 1 and V_info[v][2] == 1 and r[9] == 0 and
#
        for spec_comp in range(0, n_spec_feat):
            # Check if the current uld type has the same hazardous and temp. controlled specs as the cargo request
            this_ULD_type.append(1 if V_info[v][2 + spec_comp] >= r[8 + spec_comp] else 0)

        this_ULD_type.append(1 if V_info[v][1] >= r[6] else 0)
        if 0 not in this_ULD_type:
            # If there is no conflict between the cargo request type and the current ULD type, add the current ULD to
            # the list of possible ULDs to be generated for this cargo request.
            ULD_type_comp.append(v)
        else:
            # If there is a conflict do nothing and move to next ULD type
            pass
    # Return the list of possible ULD types for this cargo request. Eventually a random choice is made from this list
    return ULD_type_comp

U_dict = {}
for idx_r, r in R_dict.items():

    t_rt, t_dd = r[10], r[11]
    #pick_times(nodes_at_airport, r[0], r[1])

    U_dict[idx_r] = [((r[0], t_rt), (r[1], t_dd)), (N_dict_at_idx[(r[0], t_rt)],
    N_dict_at_idx[(r[1], t_dd)]), random.choice(comp_ULD_new(r, V_info, 2)), idx_r]


# U_dict = [(('ORIGIN AIRPORT', release time), ('DESTINATION AIRPORT', due date)),
#           (ORIGIN NODE, DESTINATION NODE), ULD TYPE, REQUEST IDX for which the ULD was generated]
#For every request a ULD is generated.

# Dictionary of ULDs indexed per uld type
U_uld_type     = {k:[idx for idx,u in U_dict.items() if V_info[u[2]][4]==k] for k in V_ULD_type}
U_uld_type_2     = {k:[idx for idx,u in U_dict.items() if u[2] ==k] for k in V_info}
#print(len(U_uld_type_2[0]), len(U_uld_type_2[1]), len(U_uld_type_2[2]), len(U_uld_type_2[3]))

U_uld_type_idx = {idx:key for idx,key in enumerate(U_uld_type.keys())}

U_idx_uld_type = {v:k for k,v in U_uld_type_idx.items()}

occurence_count = []
identical_ulds = {}

reduced_tuples = {}
# Check for identical values
for key, value in U_dict.items():
    reduced_tuple = tuple(value[:3])
    if reduced_tuple in reduced_tuples:
        #print(f"Keys '{reduced_tuples[reduced_tuple]}' and '{key}' have identical values.")
        #print(U_dict[reduced_tuples[reduced_tuple]], U_dict[key])
        if key not in occurence_count:
            occurence_count.append(key)
            identical_ulds[reduced_tuples[reduced_tuple]].append(key)
    else:
        reduced_tuples[reduced_tuple] = key
        identical_ulds[key] = []
identical_ulds = {k: v for k, v in identical_ulds.items() if v != []}

combined_length = 0
for key, values in identical_ulds.items():
    combined_length += len(values)
#print(len(identical_ulds))
#print(combined_length)

# For each request, determine the ULDs that can transport it. The choice is made based on the following criteria:
# - origin and destination airports should match
# - the ULD type is consistent with the request type
# - the request should not be to heavy or have a too large volume for the ULD
# - the due date of the request has to be after the release time of the ULD
# - the release time of the request has to be before the due date of the ULD

def U_r_maker():
    U_r = {}
    for r, r_values in R_dict.items():
        U_r[r] = [u for u, u_values in U_dict.items()
            if (u_values[0][0][0] == r_values[0] and u_values[0][1][0] == r_values[1] and
            comp_ULD_check(r_values, V_info, u_values[2], n_spec_feat) and r_values[5] <= V_info[u_values[2]][0] and
            r_values[6] <= V_info[u_values[2]][1] and r_values[11] >= u_values[0][0][1] and r_values[10] <= u_values[0][1][1])]
    return U_r

def subset_U_r_maker(num):
    U_r = {}
    for r, r_values in R_dict.items():
        candidates = []
        for u, u_values in U_dict.items():
            if (u_values[0][0][0] == r_values[0] and u_values[0][1][0] == r_values[1] and
            comp_ULD_check(r_values, V_info, u_values[2], n_spec_feat) and r_values[5] <= V_info[u_values[2]][0] and
            r_values[6] <= V_info[u_values[2]][1] and r_values[11] >= u_values[0][0][1] and r_values[10] <= u_values[0][1][1]):
                candidates.append(u)
        # Sort candidates by the least absolute difference and select the first xx
        sorted_candidates = sorted(
            candidates,
            key=lambda u: (abs(r_values[10] - U_dict[u][0][0][1]) + abs(r_values[11] - U_dict[u][0][1][1])))
        U_r[r] = sorted_candidates[:num]
    return U_r

if subset:
    U_r = subset_U_r_maker(max_num_of_ulds_per_R)
    # For each ULD, define the subset of requests that can be transported in that ULD
    R_u = {u: [r for r, r_values in U_r.items() if u in r_values] for u, u_values in U_dict.items()}


    # Ensure that R_u lists for any identical ULDS are always the same by making a union of all those lists
    for key, values in identical_ulds.items():
        # Get all lists corresponding to the key and its identical values
        values.append(key)
        combined_list = list(chain.from_iterable(R_u[u] for u in values if u in R_u))
        # Remove duplicates and sort
        combined_list = sorted(set(combined_list))
        # Update R_u for each ULD in the list
        for u in values:
            if u in R_u:
                R_u[u] = combined_list
        values.remove(key)
    #Update U_r accordingly
    U_r = {r: [u for u, u_values in R_u.items() if r in u_values] for r, r_values in R_dict.items()}


if not subset:
    U_r = U_r_maker()
    # For each ULD, define the subset of requests that can be transported in that ULD
    R_u = {u: [r for r, r_values in U_r.items() if u in r_values] for u, u_values in U_dict.items()}

########################################################################################################################
########################################################################################################################
########################################################################################################################
#%%
########################################################################################################################
################################################ FLEET #################################################################
########################################################################################################################
K = {0:{'type':'B747-400BCF','#':1,'PRD':[(107,7500),(63,11800),(0,14900)],
        'OC':15.23,'ULD_conf':[32,30]},
    1:{'type':'B747-400ERF','#':3,'PRD':[(112,9200),(85,11900),(0,16100)],
        'OC':16.03,'ULD_conf':[32,30]}}


N_kv = {k: K[k]['ULD_conf'] for k in K.keys()}
N_k = {k: K[k]['#'] for k in K.keys()}

# PRD = payload range diagram?
# OC = operational costs
# ULD_conf = uld space availability per type

# Importing airports
airports_IATA       = list(df_a['airport'])
airports_region     = list(df_a['region'])
airports_lat        = list(df_a['lat'])
airports_lon        = list(df_a['lon'])

def compute_max_payload(K, k, distance):
    PRD_k = K[k]['PRD']
    # Here we can fly with max. payload
    if distance <= PRD_k[0][1]:
        payload = PRD_k[0][0]
    # Here the MTOW is limiting
    elif distance <= PRD_k[1][1]:
        payload = PRD_k[0][0] * (distance - PRD_k[1][1]) / (PRD_k[0][1] - PRD_k[1][1]) + \
                  PRD_k[1][0] * (PRD_k[0][1] - distance) / (PRD_k[0][1] - PRD_k[1][1])
    # Here we are at full fuel capacity
    elif distance <= PRD_k[2][1]:
        payload = PRD_k[1][0] * (distance - PRD_k[2][1]) / (PRD_k[1][1] - PRD_k[2][1]) + \
                  PRD_k[2][0] * (PRD_k[1][1] - distance) / (PRD_k[1][1] - PRD_k[2][1])
    # If we enter this else, the distance is too much even if we were flying
    # with no payload. Hence, the direct flight is not possible in the
    # first place
    else:
        payload = -1
    return payload


# Determining maximum range across all fleet types
max_range_fleet = max([v['PRD'][-1][1] for k, v in K.items()])

from geopy.distance import great_circle
# Compute airport distance matrix (only upper triangular part), and then sum the transpose
airport_distance_matrix     = np.zeros([len(df_a),len(df_a)])
for i,row_i in df_a.iterrows():
    for j,row_j in df_a.iterrows():
        if j>i:
            airport_distance_matrix[i][j] = np.round(great_circle((row_i.lat,row_i.lon),(row_j.lat,row_j.lon)).kilometers,0)
airport_distance_matrix = airport_distance_matrix + np.transpose(airport_distance_matrix)

# Binary matrix. Unitary values if the (i,j) airport pair can be reached considering the maximum range that
# was computed above
airport_reachability_matrix = np.zeros([len(df_a),len(df_a)])
for i in range(0,len(airport_distance_matrix)):
    for j in range(0,len(airport_distance_matrix)):
        if j>i:
            airport_reachability_matrix[i][j] = 1 if airport_distance_matrix[i][j] <= max_range_fleet else 0

airport_reachability_matrix = airport_reachability_matrix + np.transpose(airport_reachability_matrix)
# Define all (i,j) pairs that are reachable via direct flight,
# store the distance, the time, and the maximum payload transportable
# depending on the fleet type
OD_pairs = []
for i,row_i in df_a.iterrows():
    for j,row_j in df_a.iterrows():
         if airport_reachability_matrix[i][j] == 1 and (i,j) not in OD_pairs:
             OD_pairs.append((i,j))
OD_pairs_dict_ij_IATA = {od:(df_a.airport[od[0]],df_a.airport[od[1]]) for od in OD_pairs}
OD_pairs_dict_IATA_ij = {v:k for k,v in OD_pairs_dict_ij_IATA.items()}

# Adding more info per (i,j) pair. For example, which aircraft type can fly this route (note that there should
# be at least one, otherwise we would not have defined this pair in the first place), and what is the maximum payload

OD_pairs_dict_ij_info = {k:{'distance':airport_distance_matrix[k[0]][k[1]],
                         'max_payload':{kk:np.round(compute_max_payload(K,kk,
                        airport_distance_matrix[k[0]][k[1]]),1)
                                        for kk in K.keys()}}
                        for k,v in OD_pairs_dict_ij_IATA.items()}

# Define for every (i,j) pair which aircraft type can fly that pair
OD_pairs_dict_ij_k_compatible = {od:[k for k in OD_pairs_dict_ij_info[od]['max_payload'].keys() if
                                     OD_pairs_dict_ij_info[od]['max_payload'][k] != -1] for od in
                                 OD_pairs_dict_ij_info.keys()}
dict_IATA_idx = {k:v for v,k in enumerate(list(df_a.airport))}
dict_idx_IATA = {v:k for k,v in dict_IATA_idx.items()}

########################################################################################################################
########################################################################################################################
########################################################################################################################
#%%
########################################################################################################################
################################################ FULL TSN NETWORK ######################################################
########################################################################################################################

# Set of all real nodes of the TSN
N_dict = {k:v for k,v in enumerate(N)}
# Set containing all (dummy) origin nodes of all ULDs: S^U (note index does not start at 0)
ULD_source_dict  = {int(k+len(N_dict)):v[0][0] for k,v in U_dict.items()}
# Set containing all (dummy) destination nodes of all ULDs: T^U (note index does not start at 0
ULD_sink_dict    = {int(k+len(N_dict)+len(ULD_source_dict)):v[0][1] for k,v in U_dict.items()}

# This is never used in the code but I guess its a set of all real and dummy nodes
N_all_dict = {**N_dict,**ULD_source_dict,**ULD_sink_dict}

# Subset of all real nodes without source and sink
N_K_realTSN = {k:v for k,v in N_dict.items() if v[0] != 'Source' and v[0] != 'Sink'}

G_dict          = {k:(N_dict_at_idx[v[0]],N_dict_at_idx[v[1]]) for k,v in G_dict_idx_at.items()}        # Set of all ground arcs
F_dict          = {k:(N_dict_at_idx[v[0]],N_dict_at_idx[v[1]]) for k,v in F_dict_idx_at.items()}        # Set of all flight arcs
Source_TSN_dict = {k:(N_dict_at_idx[v[0]],N_dict_at_idx[v[1]]) for k,v in Source_TSN_idx_at.items()}    # Set of all arcs from the source into the TSN
TSN_Sink_dict   = {k:(N_dict_at_idx[v[0]],N_dict_at_idx[v[1]]) for k,v in TSN_Sink_idx_at.items()}      # Set of all arcs from the TSN to the source

# Set of all arcs without source and sink bypass
A_all_dict_1 = {**G_dict, **F_dict, **Source_TSN_dict, **TSN_Sink_dict}

# Set of source sink bypass arcs (is really only one arc)
Source_Sink_dict = {len(A_all_dict_1):
                        (N_dict_at_idx[('Source', T_min)], N_dict_at_idx[('Sink', T_max)])}

# Set of all arcs including source sink bypass
A_all_dict_2 = {**A_all_dict_1, **Source_Sink_dict}

# Defining arcs from each ULD source to the first node in the TSN available: A^U+
ULD_source_TSN_dict = {len(A_all_dict_2) + idx:
                           (list(ULD_source_dict.keys())[idx],
                            U_dict[key][1][0])
                       for idx, key in enumerate(U_dict.keys())}

# Defining arcs from the last node in the TSN available to each ULD sink node: A^U-
TSN_ULD_sink_dict = {len(A_all_dict_2) + len(ULD_source_TSN_dict) + idx:
                         (U_dict[key][1][1], list(ULD_sink_dict.keys())[idx])
                     for idx, key in enumerate(U_dict.keys())}

# Set of all arcs including source sink bypass and arcs to dummy ULD source and sinks
A_all_dict_3 = {**A_all_dict_2, **ULD_source_TSN_dict, **TSN_ULD_sink_dict}
# Defining the subset of ground arcs that each ULD can use. We omit for now
# ground arcs whose origin is before the release time of the ULD and
# ground arcs whose destination is after the due date of the ULD


# Defining the subset of flight arcs that each ULD can use. For now, we simply cut-off the flight arcs that are  outside
# the time-range of the ULD, but we could even disregard more flight arcs (e.g., flight arcs starting at the release
# time of the ULD from airports that are not the origin airport, because the ULD cannot physically be there in such a
# short time).

F_u = {}
G_u = {}
for u, u_values in U_dict.items():
    node_paths = list(nx.all_simple_paths(G, source=u_values[1][0], target=u_values[1][1]))
    available_arcs = set()
    for path in node_paths:
        arcs = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        arc_path = [next(k for k, v in A_all_dict_2.items() if v == arc) for arc in arcs]
        available_arcs.update(arc_path)

    available_arcs = list(available_arcs)

    #print(f'For ULD {u} the available arcs are {available_arcs}')
    F_u[u] = []
    G_u[u] = []

    for arc in available_arcs:
        if arc in F_dict.keys():
            F_u[u].append(arc)
        if arc in G_dict.keys():
            G_u[u].append(arc)

average_length_Fu = sum(len(lst) for lst in F_u.values()) / len(F_u) if F_u else 0
average_length_Gu = sum(len(lst) for lst in G_u.values()) / len(G_u) if F_u else 0


# For each request, define the (source,sink) nodes where it should enter/exit the TSN if transported, currently there
# is no real requirement for this. Release dates are always the earliest node of the origin airport, due dates are
# always the latest node of the destination airport
R_dict_source_sink = {r:(N_dict_at_idx[earliest_node_airport[v[0]]],
                         N_dict_at_idx[latest_node_airport[v[1]]]) for r,v in R_dict.items()}

# Subset of arcs that aircraft can use: A^K
A_K_dict       = {**G_dict,**F_dict,
                  **Source_TSN_dict,**TSN_Sink_dict,**Source_Sink_dict}
A_K_dict_a_idx = {v:k for k,v in A_K_dict.items()}

# Subset of arcs that ULDs can use: A^U
A_U_dict       = {**G_dict,**F_dict,**ULD_source_TSN_dict,**TSN_ULD_sink_dict}
A_U_dict_a_idx = {v:k for k,v in A_U_dict.items()}

# For all types of arcs, get the indices. This will be useful when writing
# constraints later
G_arcs_idxs                  = list(G_dict.keys())
F_arcs_idxs                  = list(F_dict.keys())
Source_TSN_idxs              = list(Source_TSN_dict.keys())
TSN_sink_idxs                = list(TSN_Sink_dict.keys())
Source_Sink_dict_idxs        = list(Source_Sink_dict.keys())
ULD_source_TSN_arcs_idxs     = list(ULD_source_TSN_dict.keys())
TSN_ULD_sink_arcs_idxs       = list(TSN_ULD_sink_dict.keys())
A_all_dict_arcs_idx          = list(A_all_dict_3.keys())

# These dictionaries get their keys shifted so the index of U_dict corresponds to the index of the ULD_source_TSN arc
# and TSN_ULD_sink_dict
ULD_source_TSN_u_a_dict      = {idx_u:u for idx_u,u in enumerate(ULD_source_TSN_dict.keys())}
TSN_ULD_sink_u_a_dict        = {idx_u:u for idx_u,u in enumerate(TSN_ULD_sink_dict.keys())}

# Subset of aircraft type available for each flight arc: K_f
K_f = {f:OD_pairs_dict_ij_k_compatible[(dict_IATA_idx[N_dict[v[0]][0]],
                                        dict_IATA_idx[N_dict[v[1]][0]])]
       for f,v in F_dict.items()}

# For each ULD, define the subset of arcs that can be used: A^U_u
AU_u = {u:G_u[u]+F_u[u]+[ULD_source_TSN_u_a_dict[u]]+
        [TSN_ULD_sink_u_a_dict[u]] for u in U_dict.keys()}

# For nodes of the original TSN, define the two subsets plus and minus (resp., the subset of arcs arriving
# to or leaving from the current node). Note that these subsets are different for aircraft and ULDs as they share some
# arcs, but are also characterized by unique arcs that only aircraft/ULDs can transverse

# Subsets for aircraft. Here we need two indices because we need conservation of flow per node and per aircraft type
# Plus is arriving to the current node n
AK_k_n_plus  = {(k,n):[a for a,v in A_K_dict.items() if (a not in F_dict.keys() and n==v[1])
                        or (a in F_dict.keys() and k in K_f[a] and n==v[1])]
                        for k in K for n in N_dict.keys()}
# Minus is leaving from current node n
AK_k_n_minus = {(k,n):[a for a,v in A_K_dict.items() if (a not in F_dict.keys() and n==v[0])
                        or (a in F_dict.keys() and k in K_f[a] and n==v[0])]
                        for k in K for n in N_dict.keys()}

# Subset for ULDs. Here we need two indices because we need conservation of flow per node and per ULD
# Plus is arriving to the current node n
AU_u_n_plus  = {(u,n):[a for a in AU_u[u] if n==A_all_dict_3[a][1]]
                        for u in U_dict.keys() for n in N_dict.keys()}
# Minus is leaving from current node n
AU_u_n_minus = {(u,n):[a for a in AU_u[u] if n==A_all_dict_3[a][0]]
                        for u in U_dict.keys() for n in N_dict.keys()}
# Define subset of ULDs of type v (never used): U_v
U_v = {k:[u for u,u_info in U_dict.items() if u_info[2]==k] for k in V_info.keys()}

# Define subset of ULDs that can use a specific flight arc
U_f = {f:[u for u in U_dict.keys() if f in F_u[u]] for f in F_dict.keys()}


# It sounds like the two variables below do the same thing, they dont contain the same values though. U_vf considered
# there are 4 different ULD types instead of 2. I think 2 is the correct number of ULD types.
# Define subset of ULDs of a specific type that can use a specific flight arc
# U_vf = {(v,f):[u for u in U_f[f] if U_dict[u][2]==v] for v in V_info.keys() for f in F_dict.keys()}

# Define subset of ULDs of a specific ULD type that can use a specific flight arc
U_v_f = {(U_idx_uld_type[ULD_type],f):[u for u in U_f[f] if U_dict[u][2] in v_av] for ULD_type,v_av in V_ULD_type_idx_dict.items() for f in F_dict.keys()}

# Define for each flight arc some properties (e.g.,max payload according to
# fleet type)
F_info = {f:OD_pairs_dict_ij_info[(dict_IATA_idx[N_dict_idx_at[F_dict[f][0]][0]],
                                   dict_IATA_idx[N_dict_idx_at[F_dict[f][1]][0]])]
          for f in F_dict.keys()}
print(F_info)
W_fk = {(f,k):F_info[f]['max_payload'][k] for f in F_dict.keys() for k in K_f[f]}

# A request can be assigned to a ULD if that ULD leaves after the release time of the request, and a request can be
# assigned to a ULD if that ULD arrives before the due date of the request
# Defining subsets F_ruO and F_ruD that define respectively
# F_r_plus: set of flight arcs that ULD u can use to depart from the origin airport later than the release time
# of the request
# F_r_minus: set of flight arcs that ULD u can use to arrive at the destination airport sooner than the
# due date of the request
F_r_plus = {(r, u): [f for f in F_u[u] if ((dict_IATA_idx[N_dict_idx_at[F_dict[f][0]][0]] ==
                                         dict_IATA_idx[R_dict[r][0]]) and
                                        (N_dict_idx_at[F_dict[f][0]][1] >= R_dict[r][10]))] for r in R_dict.keys() for u in U_r[r]}


F_r_minus = {(r, u): [f for f in F_u[u] if ((dict_IATA_idx[N_dict_idx_at[F_dict[f][1]][0]] ==
                                         dict_IATA_idx[R_dict[r][1]]) and
                                        (N_dict_idx_at[F_dict[f][1]][1] <= R_dict[r][11]))] for r in R_dict.keys() for u in U_r[r]}

INC_dict_u = {
    u: {
        k: [
            (r1, r2)
            for idx_r1, r1 in enumerate(R_u[u])
            for idx_r2, r2 in enumerate(R_u[u])
            if (
                idx_r2 > idx_r1
                and (
                    (R_dict[r1][4] == k[0] and R_dict[r2][4] == k[1])
                    or (R_dict[r1][4] == k[1] and R_dict[r2][4] == k[0])
                )
            )
        ]
        for k in inc_combo
    }
    for u in U_dict.keys()
}

#%%
########################################################################################################################
############################################ DUMMY ULD GENERATION ######################################################
########################################################################################################################

dummy_U_dict = {}
dummy_U_r = {}

for idx_r, r in R_dict.items():

    t_rt, t_dd = r[10], r[11]

    dummy_U_dict[idx_r] = {'OD_tuple':((r[0], t_rt), (r[1], t_dd)),'OD_nodes_tuple':(N_dict_at_idx[(r[0], t_rt)],
    N_dict_at_idx[(r[1], t_dd)]), 'ULD_type':-1,'idx_r_original':idx_r,'Weight':r[5],'Volume':r[6]}

    #dummy_U_dict[idx_r] = [((r[0], t_rt), (r[1], t_dd)), (N_dict_at_idx[(r[0], t_rt)],
    #N_dict_at_idx[(r[1], t_dd)]), -1, idx_r, r[5], r[6]]

    dummy_U_r[idx_r] = [idx_r]
print(dummy_U_r)
# dummy_U_dict = [(('ORIGIN AIRPORT', release time), ('DESTINATION AIRPORT', due date)),
#                   (ORIGIN NODE, DESTINATION NODE), ULD TYPE, REQUEST IDX for which the ULD was generated,
#                   REQUEST WEIGHT, REQUEST VOLUME]


dummy_ULD_source_TSN_dict = {len(A_all_dict_2) + idx:
                           (list(ULD_source_dict.keys())[idx],
                            dummy_U_dict[key]['OD_nodes_tuple'][0])
                       for idx, key in enumerate(dummy_U_dict.keys())}
dummy_TSN_ULD_sink_dict = {len(A_all_dict_2) + len(ULD_source_TSN_dict) + idx:
                         (dummy_U_dict[key]['OD_nodes_tuple'][1], list(ULD_sink_dict.keys())[idx])
                     for idx, key in enumerate(dummy_U_dict.keys())}

A_all_dict_4 = A_all_dict_3 = {**A_all_dict_2, **dummy_ULD_source_TSN_dict, **dummy_TSN_ULD_sink_dict}


dummy_F_u = {}
dummy_G_u = {}
for u, u_values in dummy_U_dict.items():
    node_paths = list(nx.all_simple_paths(G, source=u_values['OD_nodes_tuple'][0], target=u_values['OD_nodes_tuple'][1]))
    available_arcs = set()
    for path in node_paths:
        arcs = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        arc_path = [next(k for k, v in A_all_dict_2.items() if v == arc) for arc in arcs]
        available_arcs.update(arc_path)

    available_arcs = list(available_arcs)

    #print(f'For ULD {u} the available arcs are {available_arcs}')
    dummy_F_u[u] = []
    dummy_G_u[u] = []

    for arc in available_arcs:
        if arc in F_dict.keys():
            dummy_F_u[u].append(arc)
        if arc in G_dict.keys():
            dummy_G_u[u].append(arc)

dummy_ULD_source_TSN_u_a_dict      = {idx_u:u for idx_u,u in enumerate(dummy_ULD_source_TSN_dict.keys())}
dummy_TSN_ULD_sink_u_a_dict        = {idx_u:u for idx_u,u in enumerate(dummy_TSN_ULD_sink_dict.keys())}

dummy_AU_u = {u:dummy_G_u[u]+dummy_F_u[u]+[dummy_ULD_source_TSN_u_a_dict[u]]+
        [dummy_TSN_ULD_sink_u_a_dict[u]] for u in dummy_U_dict.keys()}

dummy_AU_u_n_plus  = {(u,n):[a for a in dummy_AU_u[u] if n==A_all_dict_4[a][1]]
                        for u in dummy_U_dict.keys() for n in N_dict.keys()}

dummy_AU_u_n_minus = {(u,n):[a for a in dummy_AU_u[u] if n==A_all_dict_4[a][0]]
                        for u in dummy_U_dict.keys() for n in N_dict.keys()}

dummy_U_f = {f:[u for u in dummy_U_dict.keys() if f in dummy_F_u[u]] for f in F_dict.keys()}

dummy_F_r_plus = {(r, u): [f for f in dummy_F_u[u] if ((dict_IATA_idx[N_dict_idx_at[F_dict[f][0]][0]] ==
                                         dict_IATA_idx[R_dict[r][0]]) and
                                        (N_dict_idx_at[F_dict[f][0]][1] >= R_dict[r][10]))] for r in R_dict.keys() for u in dummy_U_r[r]}

dummy_F_r_minus = {(r, u): [f for f in dummy_F_u[u] if ((dict_IATA_idx[N_dict_idx_at[F_dict[f][1]][0]] ==
                                         dict_IATA_idx[R_dict[r][1]]) and
                                        (N_dict_idx_at[F_dict[f][1]][1] <= R_dict[r][11]))] for r in R_dict.keys() for u in dummy_U_r[r]}

########################################################################################################################
########################################################################################################################
########################################################################################################################


end_time = time.time()
elapsed_time = end_time - start_time

# Print the runtime in seconds
print(f"Elapsed preprocessing time: {elapsed_time:.4f} seconds")
print("Nodes", N_dict)

print("Ground arcs", G_dict)
print("Flight arcs", F_dict)
print("Source -> TSN arcs", Source_TSN_dict)
print("TSN -> Sink arcs", TSN_Sink_dict)
print("Source -> Sink arcs", Source_Sink_dict)
print("ULD Source -> TSN arcs", ULD_source_TSN_dict)
print("TSN -> ULD Sink arcs", TSN_ULD_sink_dict)

print("Fleet",K)
print(len(U_dict),"ULDs")

#print(F_info)
if schedule_small:
    print('USING SMALL SCHEDULE!')
if not schedule_small:
    print('USING LARGE SCHEDULE!')

if clustering_requests:
    if clustering_maximum == 0:
        print('CLUSTERING REQUESTS WITH A MAXIMUM SIZE OF THE SMALL ULD PER CLUSTER!')
    if clustering_maximum == 1:
        print('CLUSTERING REQUESTS WITH A MAXIMUM SIZE OF THE LARGE ULD PER CLUSTER!')
if not clustering_requests:
    print('NOT CLUSTERING REQUESTS')

if subset:
    print(f'USING A MAXIMUM NUMBER OF {max_num_of_ulds_per_R} ULDs PER REQUEST!')
if not subset:
    print('NO LIMITATION ON ULDs PER REQUEST')

# Calculate the total length of all R_u[u] and the number of u in U_dict
total_length = sum(len(R_u[u]) for u in U_dict)
number_of_u = len(U_dict)

# Calculate the average length
average_length = total_length / number_of_u if number_of_u > 0 else 0

# Calculate the maximum length
max_length = max(len(R_u[u]) for u in U_dict) if number_of_u > 0 else 0

# Output the results
print(f"Average length of R_u over all u in U_dict: {average_length}")
print(f"Maximum length of R_u over all u in U_dict: {max_length}")



#%%
########################################################################################################################
############################################### VISUAL PLOT ############################################################
########################################################################################################################

# Extract data for plotting (excluding 'Source' and 'Sink' nodes)
airport_names = []
times = []
node = []
annotations = []

if schedule_small:
    plt.figure(figsize=(10, 6))
if not schedule_small:
    plt.figure(figsize=(14, 8.4))

adjustment_dict_full = {
    0: (3,10),
    1: (0,10),
    2: (0,10),
    3: (0,10),
    4: (0,10),
    5: (0,10),
    6: (0,10),
    7: (-2,10),
    8: (2,10),
    9: (0,10),
    10: (4.5,10),
    11: (-3,10),
    12: (3,10),
    13: (-3,10),
    14: (3,10),
    15: (-3,10),
    16: (3,10),
    17: (-4,10),
    18: (4,10),
    19: (0,10),
    20: (4,10),
    21: (-3,10),
    22: (3,10),
    23: (-3,10),
    24: (3,10),
    25: (-3,10),
    26: (3,10),
    27: (-13,10),
    28: (6,10),
    29: (-5,10),
    30: (0,10),
    31: (-3,10),
    32: (3,10),
    33: (-4,10),
    34: (4,10),
    35: (-4,10),
    36: (4,10),
    37: (0,10),
    38: (5,10),
    39: (-3,10),
    40: (3,10),
    41: (-3,10),
    42: (3,10),
    43: (-3,10),
    44: (3,10),
    45: (-3,10),
    46: (3,10),
    47: (-3,10),
    48: (3,10),
    49: (0,10),
    50: (0,10),
    51: (-3,10),
    52: (3,10),
    53: (-3,10),
    54: (3,10),
    55: (-4,10),
    56: (15,10),
    57: (5,10),
    58: (0,10),
    59: (-3,10),
    60: (3,10),
    61: (0,10),
    62: (0,10),
    63: (0,10),
    64: (-3,10),
    65: (0,10),
    66: (0,10),
    67: (0,10),
    68: (0,10),
    69: (0,10),
    70: (0,10),
    71: (-3,10),
    72: (3,10),
    73: (-4,10),
    74: (4,10),
    75: (0,10),
    76: (0,10),
    77: (-5,10),
    78: (-4,10),
    79: (0,10),
    80: (0,10),
    81: (-4,10),
    82: (4,10),
    83: (-3,10),
    84: (3,10),
    85: (0,10),
    86: (0,10),
    87: (0,10),
    88: (0,10),
    89: (-4,10),
    90: (4,10),
    91: (-3,10),
    92: (3,10),
    93: (0,10)
}
adjustment_dict_red = {
    0: (0,10),
    1: (0,10),
    2: (0,10),
    3: (0,10),
    4: (0,10),
    5: (0,10),
    6: (0,10),
    7: (0,10),
    8: (3,10),
    9: (-1,10),
    10: (0,10),
    11: (0,10),
    12: (0,10),
    13: (0,10),
    14: (0,10),
    15: (0,10),
    16: (3,10),
    17: (0,10),
    18: (0,10),
    19: (0,10),
    20: (0,10),
    21: (0,10),
    22: (0,10),
    23: (0,10),
    24: (0,10),
    25: (0,10),
    26: (0,10),
    27: (0,10),
    28: (0,10),
    29: (0,10),
    30: (0,10),
    31: (0,10),
}
for idx, (airport, time) in N_dict.items():
    if airport not in ['Source', 'Sink']:
        airport_names.append(airport)
        time_stamp = datetime.fromtimestamp(time, tz=timezone.utc)
        times.append(time_stamp)
        node.append(idx)
        # Plot point
        plt.scatter(time_stamp, airport, color='blue', marker='o')
        # Annotate point with index
        #plt.text(time_stamp, airport, str(idx), ha='center', va='bottom')
        # Collect annotations to adjust later
        if schedule_small:
            annotation = plt.annotate(
                str(idx),
                (time_stamp, airport),
                textcoords="offset points",
                xytext=adjustment_dict_red[idx],  # Initial offset, will be adjusted
                ha='center',
                fontsize=8,  # Adjust the fontsize for readability
                bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='#acbcfc', alpha=0.7)
                # Background for better readability
            )
            annotations.append(annotation)
        if not schedule_small:
            annotation = plt.annotate(
                str(idx),
                (time_stamp, airport),
                textcoords="offset points",
                xytext=adjustment_dict_full[idx],  # Initial offset, will be adjusted
                ha='center',
                fontsize=8,  # Adjust the fontsize for readability
                bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='#acbcfc', alpha=0.7)
                # Background for better readability
            )
            annotations.append(annotation)

ax = plt.gca()

if schedule_small:
    start_limit = datetime(2024, 6, 24, 12, 0)
    end_limit = datetime(2024, 6, 27, 6, 0)
    ax.set_xlim(start_limit, end_limit)

if not schedule_small:
    start_limit = datetime(2024, 6, 24, 6, 0)
    end_limit = datetime(2024, 7, 2, 12, 0)
    ax.set_xlim(start_limit, end_limit)

ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %d-%m-%y \n %H:%M'))

ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))  # Minor ticks every 6 hours
ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))  # Format for minor ticks

# Add extra major tick at the start_limit
# Convert start_limit to float for comparison and tick setting
start_limit_float = mdates.date2num(start_limit)
current_ticks = ax.get_xticks()

# Check if start_limit is already in current_ticks
if start_limit_float not in current_ticks:
    current_ticks = list(current_ticks)  # Convert to list if it's a NumPy array
    current_ticks.insert(0, start_limit_float)  # Insert start_limit as the first tick
    current_ticks = sorted(current_ticks)  # Sort the tick positions

    # Update the x-ticks with the new ticks
    ax.set_xticks(current_ticks)

plt.setp(ax.get_xticklabels(which='major'), rotation=45, ha='right', rotation_mode='anchor')
plt.setp(ax.get_xticklabels(which='minor'), rotation=45, ha='center')

# Plotting
#plt.scatter(times, airport_names, color='blue', marker='o')
plt.xlabel('Time')
plt.ylabel('Airport')
#plt.title('Visual representation of the Time-Space Network')
ax.grid(True, linestyle='--', color='#D3D3D3', linewidth=0.6)  # Light grey grid with width 0.7

plt.tight_layout()

for idx, (start, end) in G_dict.items():
    plt.plot([datetime.fromtimestamp(N_dict[start][1], tz=timezone.utc),
             datetime.fromtimestamp(N_dict[end][1], tz=timezone.utc)],
             [N_dict[start][0], N_dict[end][0]], color='grey', linewidth=1)

for idx, (start, end) in F_dict.items():
    plt.plot([datetime.fromtimestamp(N_dict[start][1], tz=timezone.utc),
             datetime.fromtimestamp(N_dict[end][1], tz=timezone.utc)],
            [N_dict[start][0], N_dict[end][0]], color='#e88900')


for idx, (airport, time) in N_dict.items():
    if airport not in ['Source', 'Sink']:
        airport_names.append(airport)
        time_stamp = datetime.fromtimestamp(time, tz=timezone.utc)
        times.append(time_stamp)
        node.append(idx)
        plt.scatter(time_stamp, airport, color='blue', marker='o')

# Adjust text to avoid overlap
#adjust_text(annotations)

#plt.savefig('time_space_network.pdf', format='pdf')
#plt.savefig('time_space_network_reducedschedule.svg', format='svg')
#plt.savefig('time_space_network.eps', format='eps')


#%%
########################################################################################################################
##################################### VISUAL REQUESTS AVAILABLE PER ULD ################################################
########################################################################################################################

# Initialize lists to store the percentages and users
requests = []
ULDs = []
for u in U_dict.keys():
    requests.append(len(R_u[u]))
    #print(u, U_dict[u])
    ULDs.append(u)

# Plot the percentages as a bar chart
plt.figure(figsize=(8, 8*0.6))
plt.bar(ULDs, requests, color='skyblue', width=1)
plt.xlabel('ULDs')
plt.xlim(-200,4000)
plt.ylim(0,355)
plt.ylabel('Requests available')
plt.title('Requests available by ULD Full Schedule')
plt.xticks(rotation=45)
plt.tight_layout()

#plt.savefig('R_u_barplot_fullschedule.svg', format='svg')

if __name__ == "__main__":
    plt.show()

