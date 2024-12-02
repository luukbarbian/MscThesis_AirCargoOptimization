import pickle
import setsparameters
import inputs
from setsparameters import comp_ULD_check
import numpy as np
from collections import defaultdict
import logging



F_dict = setsparameters.F_dict
U_uld_type = setsparameters.U_uld_type
U_v_f = setsparameters.U_v_f
U_dict = setsparameters.U_dict
A_all_dict_2 = setsparameters.A_all_dict_2
F_u = setsparameters.F_u
R_dict = setsparameters.R_dict
K = setsparameters.K
airports_IATA = setsparameters.airports_IATA
V_info = setsparameters.V_info
F_info = setsparameters.F_info
R_u = setsparameters.R_u
W_fk = setsparameters.W_fk

INC_dict_u = setsparameters.INC_dict_u

name = ''
if inputs.method == 'CG':
    if inputs.schedule_small:
        name += 'reduced'
    if not inputs.schedule_small:
        name +=  'full'
    name = name + '_' + inputs.comp_level
    name = name + '_' + str(inputs.Payload)
    if inputs.free_aircraft_placement:
        name = name + '_free'
    if not inputs.free_aircraft_placement:
        name = name + '_restricted'
    if inputs.subset:
        name = name + '_subset' + str(inputs.max_num_of_ulds_per_R)
    if inputs.clustering_requests:
        if inputs.clustering_maximum == 0:
            name = name + '_clusteringSmall'
        if inputs.clustering_maximum == 1:
            name = name + '_clusteringLarge'
    name = name + '_' + str(inputs.number_of_iterations) + 'it'
    if inputs.linearize_PP:
        name = name + '_linPP'
    if inputs.w_min:
        name = name + '_w_min'
    if inputs.w_max:
        name = name + '_w_max'
    if inputs.paths_per_request_cap:
        name = name + '_cap' + str(inputs.max_num_paths_per_r)
    if inputs.diversification:
        name = name + '_div' + inputs.div_method
    name = name + '_seed' + str(inputs.r_seed) + '_CG'

if inputs.method == 'SEQ':
    if inputs.schedule_small:
        name = 'reduced'
    if not inputs.schedule_small:
        name = 'full'
    name = name + '_' + inputs.comp_level
    name = name + '_' + str(inputs.Payload)
    if inputs.subset:
        name = name + '_subset' + str(inputs.max_num_of_ulds_per_R)
    if inputs.clustering_requests:
        if inputs.clustering_maximum == 0:
            name = name + '_clusteringSmall'
        if inputs.clustering_maximum == 1:
            name = name + '_clusteringLarge'
    if inputs.free_aircraft_placement:
        name = name + '_free'
    if not inputs.free_aircraft_placement:
        name = name + '_restricted'
    name = name + '_seed' + str(inputs.r_seed) + '_SEQ'

if inputs.method == 'AB':
    if inputs.schedule_small:
        name = 'reduced'
    if not inputs.schedule_small:
        name = 'full'
    name = name + '_' + inputs.comp_level
    name = name + '_' + str(inputs.Payload)
    if inputs.subset:
        name = name + '_subset' + str(inputs.max_num_of_ulds_per_R)
    if inputs.clustering_requests:
        if inputs.clustering_maximum == 0:
            name = name + '_clusteringSmall'
        if inputs.clustering_maximum == 1:
            name = name + '_clusteringLarge'
    if inputs.free_aircraft_placement:
        name = name + '_free'
    if not inputs.free_aircraft_placement:
        name = name + '_restricted'
    name = name + '_seed' + str(inputs.r_seed) + '_AB'

# Override name manually:
#name = 'full_L_50_subset15_restricted_seed42_SEQ'
print(f'Name of analysed instance: {name}')

# Specify the file names
x_file_name = 'pickle_files/'+name+'/x_data.pickle'
y_file_name = 'pickle_files/'+name+'/y_data.pickle'
z_file_name = 'pickle_files/'+name+'/z_data.pickle'


## Configure logging
logging.basicConfig(
    level=logging.INFO,  # Log messages with level INFO or higher
    format='%(asctime)s - %(levelname)s - %(message)s',  # Format of log messages
    handlers=[
        logging.FileHandler("logs/solution_analyser/"+name+"_solution.log"),  # Write logs to a file named output.log
        logging.StreamHandler()  # Also output logs to the console
    ]
)
if "CG" in name:
    p_file_name = 'pickle_files/'+name+'/P_dict.pickle'

    with open(p_file_name, "rb") as file:
        P_dict = pickle.load(file)
        C_up = pickle.load(file)
        R_up = pickle.load(file)
        W_up = pickle.load(file)
        paths_per_it_dict = pickle.load(file)
        elapsed_time_global = pickle.load(file)

    print('Numer of paths:',len(P_dict.keys()))
    print('CG time:', elapsed_time_global)
    print(paths_per_it_dict)

    logging.info(f'Numer of paths: {len(P_dict.keys())}')
    logging.info(f'CG time: {elapsed_time_global}')
    logging.info(paths_per_it_dict)

# Write the dictionaries to pickle files
with open(x_file_name, "rb") as file:
    x_dict = pickle.load(file)

with open(y_file_name, "rb") as file:
    y_dict = pickle.load(file)
    if "CG" in name:
        PATH_uld_routing_costs = pickle.load(file)

with open(z_file_name, "rb") as file:
    z_dict = pickle.load(file)
    if "CG" in name:
        PATH_revenue = pickle.load(file)


weight_counter = defaultdict(int)
volume_counter = defaultdict(int)

uld_lf = []
uld_vf = []
for u in U_dict.keys():
    if U_dict[u][2] in (0, 1):
        uld_weight = 1.5
        uld_volume = 4.5
    elif U_dict[u][2] in (2, 3):
        uld_weight = 4.6
        uld_volume = 10.7
    for r in R_dict.keys():
        value = z_dict.get((r, u))
        if value:
            weight_counter[u] += R_dict[r][5]
            volume_counter[u] += R_dict[r][6]
    if weight_counter[u] > 0:
        uld_lf.append(weight_counter[u] / uld_weight)
        uld_vf.append(volume_counter[u] / uld_volume)
    #if weight_counter[u] > uld_weight:
    #    print(f'WATCH OUT! ULD {u} has a weight of {weight_counter[u]} while only a capacity of {uld_weight}')
    #if volume_counter[u] > uld_volume:
    #    print(f'WATCH OUT! ULD {u} has a volume of {volume_counter[u]} while only a capacity of {uld_volume}')
F_plr_lf = {}
flight_costs = 0
ULDs_on_flight = {}
R_on_f = {}
for f in F_dict.keys():
    ULDs_on_flight[f] = []
    found = False  # To track if any k exists for the current f
    for k in K.keys():
        if x_dict.get((f, k)) is not None and x_dict.get((f, k)) >= 0.9:

            LD3_count = 0
            LD7_count = 0
            weight = 0

            for u in U_dict.keys():
                if y_dict.get((f,u)) is not None and y_dict.get((f, u)) >= 0.9:
                    ULDs_on_flight[f].append(u)
                    if U_dict[u][2] in (0, 1):
                        LD3_count += 1
                    if U_dict[u][2] in (2, 3):
                        LD7_count += 1

            R_on_f[f] = [r for r in R_dict.keys() for u in ULDs_on_flight[f] if
                             z_dict.get((r, u)) is not None and z_dict.get((r, u)) >= 0.9]
            for r in R_on_f[f]:
                weight += R_dict[r][5]

            F_plr_lf[f] = weight / W_fk[(f, k)]

            print(f'Flight arc {f} is being flown by aricraft type {k} and '
                  f'has {LD3_count} LD3 units and {LD7_count} LD7 units,'
                  f'and a payload range based load factor of {F_plr_lf[f]}')
            logging.info(f'Flight arc {f} is being flown by aricraft type {k} and '
                  f'has {LD3_count} LD3 units and {LD7_count} LD7 units,'
                  f'and a payload range based load factor of {F_plr_lf[f]}')
            #print(f'Flight arc {f} is being flown by aircraft type {k}')
            #logging.info(f'Flight arc {f} is being flown by aircraft type {k}')

            flight_costs -= K[k]['OC'] * F_info[f]['distance']
            found = True
    if not found:
        print(f'FLIGHT ARC {f} IS NOT BEING FLOWN BY ANY AIRCRAFT TYPE')
        logging.info(f'FLIGHT ARC {f} IS NOT BEING FLOWN BY ANY AIRCRAFT TYPE')

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
combined_weight =0
for r in R_transported:
     combined_weight += R_dict[r][5]
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

R_no_feature_transported = [idx for idx in R_transported if R_dict[idx][8] == 0 and R_dict[idx][9] == 0]
R_hazfeature_transported = [idx for idx in R_transported if R_dict[idx][8] == 1 and R_dict[idx][9] == 0]
R_tempfeature_transported = [idx for idx in R_transported if R_dict[idx][9] == 1 and R_dict[idx][8] == 0]
R_bothfeature_transported = [idx for idx in R_transported if R_dict[idx][8] == 1 and R_dict[idx][9] == 1]

total_chemical_count = sum(1 for key, value in R_dict.items() if value[4] == 'chemical')
transported_chemical_count = sum(1 for key in R_dict if key in R_transported and R_dict[key][4] == 'chemical')
percentage_chemical = transported_chemical_count / total_chemical_count
print(f'Percentage of chemical requests transported: {percentage_chemical}')

total_perishable_count = sum(1 for key, value in R_dict.items() if value[4] == 'perishable')
transported_perishable_count = sum(1 for key in R_dict if key in R_transported and R_dict[key][4] == 'perishable')
percentage_perishable = transported_perishable_count / total_perishable_count
print(f'Percentage of perishable requests transported: {percentage_perishable}')

total_heavy_count = sum(1 for key, value in R_dict.items() if value[4] == 'heavy')
transported_heavy_count = sum(1 for key in R_dict if key in R_transported and R_dict[key][4] == 'heavy')
percentage_heavy = transported_heavy_count / total_heavy_count
print(f'Percentage of heavy requests transported: {percentage_heavy}')

total_other_count = sum(1 for key, value in R_dict.items() if value[4] == 'other')
transported_other_count = sum(1 for key in R_dict if key in R_transported and R_dict[key][4] == 'other')
percentage_other = transported_other_count / total_other_count
print(f'Percentage of other requests transported: {percentage_other}')

ULDs_on_arc = {}
ULDs_OD = {}
ULD_used = []
# Iterate through the original dictionary
for (arc, u), value in y_dict.items():
    # If the arc is not already in the dictionary, add it with an empty list
    if arc not in ULDs_on_arc:
        ULDs_on_arc[arc] = []
    # Append the ULD to the list of ULDs for this arc
    ULDs_on_arc[arc].append(u)
    if (U_dict[u][0][0][0], U_dict[u][0][1][0]) not in ULDs_OD:
        ULDs_OD[(U_dict[u][0][0][0], U_dict[u][0][1][0])] = []
    if u not in ULDs_OD[(U_dict[u][0][0][0], U_dict[u][0][1][0])]:
        ULDs_OD[(U_dict[u][0][0][0], U_dict[u][0][1][0])].append(u)
    if u not in ULD_used:
        ULD_used.append(u)
ratio_ULD_used = len(ULD_used)/len(U_dict) * 100
U_uld_type_used   = {k:[idx for idx in ULD_used if U_dict[idx][2] ==k] for k in V_info}


Requests_in_ULD = {}
for (r, u), value in z_dict.items():
    # If the arc is not already in the dictionary, add it with an empty list
    if u not in Requests_in_ULD:
        Requests_in_ULD[u] = []
    # Append the ULD to the list of ULDs for this arc
    Requests_in_ULD[u].append(r)

R_in_too_advanced_ULD = []
for u in U_uld_type_used[1]:
    if u in Requests_in_ULD.keys():
        for r in R_no_feature_transported:
            if r in Requests_in_ULD[u]:
                R_in_too_advanced_ULD.append(r)

        #condition = True
        #for r in Requests_in_ULD[u]:
        #    if r in R_tempfeature_transported:
        #        condition = False
        #    if r in R_hazfeature_transported:
        #        condition = False
        #    if r in R_bothfeature_transported:
        #        condition = False
        #if condition:
        #    print(f'NONE OF THE REQUESTS IN ULD {u} OF TYPE LD3+ NEED TEMP CONTROL')
        #   #for r in Requests_in_ULD[u]:
        #   #    print(u, r, R_dict[r])



for u in U_uld_type_used[3]:
    if u in Requests_in_ULD.keys():
        for r in R_no_feature_transported:
            if r in Requests_in_ULD[u]:
                R_in_too_advanced_ULD.append(r)

        #condition = True
        #for r in Requests_in_ULD[u]:
        #    if r in R_tempfeature_transported:
        #        condition = False
        #    if r in R_hazfeature_transported:
        #        condition = False
        #    if r in R_bothfeature_transported:
        #        condition = False
        #if condition:
        #    print(f'NONE OF THE REQUESTS IN ULD {u} OF TYPE LD7+ NEED TEMP CONTROL')

R_nofeature_advancedULD_ratio = len(R_in_too_advanced_ULD)/len(R_no_feature_transported) * 100 if len(R_no_feature_transported) > 0 else -1

average_uld_unit_cost = (len(U_uld_type_used[0]) * 0.105 + len(U_uld_type_used[1]) * 0.115 + len(U_uld_type_used[2]) * 0.1 + len(U_uld_type_used[3]) * 0.135 )/ len(ULD_used)

uld_routing_costs = 0
for f in F_dict.keys():
    ULDs_to_try = ULDs_on_arc.get(f)
    if ULDs_to_try == None:
        continue
    for u in ULDs_to_try:
        uld_type = U_dict[u][2]
        uld_routing_costs -= V_info[uld_type][5] * F_info[f]['distance']
        #uld_routing_costs -= 0.1 * F_info[f]['distance']

print(f'Revenue with transported requests: {revenue}')
logging.info(f'Revenue with transported requests: {revenue}')

if 'CG' in name:
    print(f'Path based revenue: {PATH_revenue}')
    logging.info(f'Path based revenue: {PATH_revenue}')

print(f'Costs of the flights operated: {flight_costs}')
print(f'Costs of routing the ULDs {uld_routing_costs}')

logging.info(f'Costs of the flights operated: {flight_costs}')
logging.info(f'Costs of routing the ULDs {uld_routing_costs}')

if 'CG' in name:
    print(f'Path based ULD routing costs: {PATH_uld_routing_costs}')
    logging.info(f'Path based ULD routing costs: {PATH_uld_routing_costs}')

print(f'Ratio of used ULDs {ratio_ULD_used}')
print(f'Number of ULD type 0 used: {len(U_uld_type_used[0])}')
print(f'Number of ULD type 1 used: {len(U_uld_type_used[1])}')
print(f'Number of ULD type 2 used: {len(U_uld_type_used[2])}')
print(f'Number of ULD type 3 used: {len(U_uld_type_used[3])}')
print(f'Total number of ULDs used: {len(ULD_used)}')
print(f'Average ULD unit cost: {average_uld_unit_cost}')
print(f'Average load factor per ULD: {sum(uld_lf)/len(uld_lf)}')
print(f'Average volume factor per ULD: {sum(uld_vf)/len(uld_vf)}')
print(f'Average payload range based load factor over all flights: {sum(F_plr_lf.values()) / len(F_plr_lf)}')


logging.info(f'Ratio of used ULDs {ratio_ULD_used}')
logging.info(f'Number of ULD type 0 used: {len(U_uld_type_used[0])}')
logging.info(f'Number of ULD type 1 used: {len(U_uld_type_used[1])}')
logging.info(f'Number of ULD type 2 used: {len(U_uld_type_used[2])}')
logging.info(f'Number of ULD type 3 used: {len(U_uld_type_used[3])}')
logging.info(f'Total number of ULDs used: {len(ULD_used)}')
logging.info(f'Average ULD unit cost: {average_uld_unit_cost}')
logging.info(f'Average load factor per ULD: {sum(uld_lf)/len(uld_lf)}')
logging.info(f'Average volume factor per ULD: {sum(uld_vf)/len(uld_vf)}')
logging.info(f'Average payload range based load factor over all flights: {sum(F_plr_lf.values()) / len(F_plr_lf)}')

print(f'Estimated objective function {revenue+flight_costs+uld_routing_costs}')
print(f'Number of transported requests: {len(R_transported)}')
print(f'Number of total requests: {len(R_dict)}')
print(f'Average revenue per request of transported requests: {average_rev_transported}')
print(f'Average revenue per request of unfulfilled requests: {average_rev_unfulfilled}')
print(f'Revenue per transported tonne: {revenue/combined_weight}')
print(f'Total transported tonnes: {combined_weight}')


logging.info(f'Estimated objective function {revenue+flight_costs+uld_routing_costs}')
logging.info(f'Number of transported requests: {len(R_transported)}')
logging.info(f'Number of total requests: {len(R_dict)}')
logging.info(f'Average revenue per request of transported requests: {average_rev_transported}')
logging.info(f'Average revenue per request of unfulfilled requests: {average_rev_unfulfilled}')
logging.info(f'Revenue per transported tonne: {revenue/combined_weight}')
logging.info(f'Total transported tonnes: {combined_weight}')

print(f'Number of transported requests without feature being transported: {len(R_no_feature_transported)}')
print(f'Number of transported hazardous requests (not temp): {len(R_hazfeature_transported)}')
print(f'Number of transported temp sens requests (not haz): {len(R_tempfeature_transported)}')
print(f'Number of both hazardous and temp sens requests being transported: {len(R_bothfeature_transported)}')
print(f'Ratio of no feature request in a too advanced ULD: {R_nofeature_advancedULD_ratio}')
print(f'Total ratio of requests transported: {R_transported_ratio}')

#print(f'Request ids per OD pair: {R_OD}')
#print(f'unfulfilled request ids per OD pair: {R_OD_unfulfilled}')
print(f'Ratio of fulfilled requests per OD pair {R_OD_ratio}')

logging.info(f'Number of transported requests without feature being transported: {len(R_no_feature_transported)}')
logging.info(f'Number of transported hazardous requests (not temp): {len(R_hazfeature_transported)}')
logging.info(f'Number of transported temp sens requests (not haz): {len(R_tempfeature_transported)}')
logging.info(f'Number of both hazardous and temp sens requests being transported: {len(R_bothfeature_transported)}')
logging.info(f'Ratio of no feature request in a too advanced ULD: {R_nofeature_advancedULD_ratio}')
logging.info(f'Total ratio of requests transported: {R_transported_ratio}')
logging.info(f'Ratio of fulfilled requests per OD pair {R_OD_ratio}')
