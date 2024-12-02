import subprocess
import sys

# Function to update inputs.py with new parameters for each experiment
def update_inputs(method, schedule_small, clustering_requests, clustering_maximum, subset, max_num_of_ulds_per_R, comp_level,
                  Payload, free_aircraft_placement, r_seed, linearize_PP, w_max, w_min, number_of_iterations,
                  paths_per_request_cap, max_num_paths_per_r, number_of_solutions, number_of_solutions_div, diversification, div_method):
    inputs_content = f"""
# inputs.py

# General settings
method = '{method}'
schedule_small = {schedule_small}
clustering_requests = {clustering_requests}
clustering_maximum = {clustering_maximum}
subset = {subset}
max_num_of_ulds_per_R = {max_num_of_ulds_per_R}
comp_level = '{comp_level}'
Payload = {Payload}
free_aircraft_placement = {free_aircraft_placement}
r_seed = {r_seed}

# RMP_CG_classes.py specific settings
linearize_PP = {linearize_PP}
w_max = {w_max}
w_min = {w_min}
number_of_iterations = {number_of_iterations}
paths_per_request_cap = {paths_per_request_cap}
max_num_paths_per_r = {max_num_paths_per_r}
number_of_solutions = {number_of_solutions}
number_of_solutions_div = {number_of_solutions_div}
diversification = {diversification}
div_method = '{div_method}'  # all or any
"""
    with open('inputs.py', 'w') as f:
        f.write(inputs_content)

# Function to run the main script
def run_CG_script():
    subprocess.run([sys.executable, "RMP_CG_classes.py"])

def run_SEQ_script():
    subprocess.run([sys.executable, "sequential_stage2.py"])

def run_AB_script():
    subprocess.run([sys.executable, "arcbased.py"])

# Define different sets of inputs for each experiment
experiments_CG = [
{
        'method': 'CG', 'schedule_small': False, 'clustering_requests': False, 'clustering_maximum': 1, 'subset': True,
        'max_num_of_ulds_per_R': 15, 'comp_level': 'H', 'Payload': 65, 'free_aircraft_placement': True, 'r_seed': 42,
        'linearize_PP': False, 'w_max': True, 'w_min': False, 'number_of_iterations': 5,
        'paths_per_request_cap': True, 'max_num_paths_per_r': 200, 'number_of_solutions': 20,
        'number_of_solutions_div': 35,
        'diversification': False, 'div_method': 'all'
    },
]

experiments_SEQ = [
    {
        'method': 'SEQ', 'schedule_small': False, 'clustering_requests': False, 'clustering_maximum': 1, 'subset': True,
        'max_num_of_ulds_per_R': 15, 'comp_level': 'H', 'Payload': 65, 'free_aircraft_placement': True, 'r_seed': 42,
        'linearize_PP': False, 'w_max': True, 'w_min': False, 'number_of_iterations': 5,
        'paths_per_request_cap': True, 'max_num_paths_per_r': 200, 'number_of_solutions': 20,
        'number_of_solutions_div': 35,
        'diversification': False, 'div_method': 'all'
    },
]

experiments_AB = [
    {
        'method': 'AB', 'schedule_small': False, 'clustering_requests': False, 'clustering_maximum': 0, 'subset': True,
        'max_num_of_ulds_per_R': 15, 'comp_level': 'L', 'Payload': 35, 'free_aircraft_placement': True, 'r_seed': 42,
        'linearize_PP': False, 'w_max': True, 'w_min': False, 'number_of_iterations': 5,
        'paths_per_request_cap': True, 'max_num_paths_per_r': 200, 'number_of_solutions': 20,
        'number_of_solutions_div': 35,
        'diversification': False, 'div_method': 'all'
    },

]


for experiment in experiments_CG:
    # Update inputs.py with parameters from the current experiment
    update_inputs(**experiment)

    # Run the main script
    run_CG_script()

for experiment in experiments_SEQ:
    # Update inputs.py with parameters from the current experiment
    update_inputs(**experiment)

    # Run the main script
    run_SEQ_script()

for experiment in experiments_AB:
    # Update inputs.py with parameters from the current experiment
    update_inputs(**experiment)

    # Run the main script
    run_AB_script()






