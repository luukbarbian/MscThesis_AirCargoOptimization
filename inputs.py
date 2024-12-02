# inputs.py

# General settings
method = 'CG'
schedule_small = False
clustering_requests = False
clustering_maximum = 1
subset = True
max_num_of_ulds_per_R = 15
comp_level = 'H'
Payload = 50
free_aircraft_placement = True
r_seed = 42

# RMP_CG_classes.py specific settings
linearize_PP = False
w_max = True
w_min = False
number_of_iterations = 5
paths_per_request_cap = True
max_num_paths_per_r = 200
number_of_solutions = 20
number_of_solutions_div = 35
diversification = False
div_method = 'all'  # all or any
