#TODO Rewrite it with Dictionary
## Powerflow simulation setting
# Load factor
load_factor = 1 # what is the reason that LF is influncing B estimation? Note this is effecting nr and ni.
load_factor_list = []

#Slack bus where (can be replaced by checking Bus Type)

# slack_bus_num = 69 
# slack_bus_num = 11
"unknwon_branch setting"

unknown_branch_G = False
unknown_G_branch_id = [5]

unknown_branch_B = False
# unknown_B_branch_id = [103] # An interesting bus for 118bus case to check
# unknown_B_branch_id = [16, 17, 18, 19, 30] #Design good SBT test for 118
# unknown_B_branch_id = [18, 20, 23, 26, 30] #What is the maximum number of unkwnon?
# unknown_B_branch_id = list(range(0, 40))
unknown_B_branch_id = [175]

delta_B = False
get_estB = False
B_ineq_cosntraint = True # No need for McCormick

unknown_branch_sh = False
unknown_sh_branch_id = [0]

unknown_transformer_tr = False
unknown_tr_transformer_id = [0]

"IPOPT setting"

"Set error matrix"
w_slack = 1
w_nv    = 1
w_nr    = 1
w_ni    = 1 # This one effects B the most

origional_code = False

#Tolerance seting 
tolerance = 1E-8

# Max iteration step
max_iter = 10000

# tee set to show detailed iteration steps
flag_tee = False

# Set objective scaling factor
Obj_scal_factor = 1e-9 # This will freak out McCormick when it is 1e20 or 1e-15, why, shouldn't it be convex?
# Obj_scal_factor = 1
# If 
G_noise = False # Now assuming noise also have such lower and upper bounds

model_type = 4
## Convex relaxation setting 1 to 3 are 3 models, type 4 is for test only

# Control if use convex relaxation (only True when unknown branch exists)
flag_McCormick = True
test_Mc = True

if flag_McCormick:
	# Keep the non_linear voltage magnitue or not
	non_convex_equality = True

	# If SBT on B is used
	tighten_B = True

	"Other Convex relaxations"
	qc_relaxation = False
	cone_relaxation = True
	B_ineq_cosntraint = True
else:
	non_convex_equality = False
	tighten_B = False
	qc_relaxation = False
	cone_relaxation = False

"Measurement setting"

flag_noise = True #Do we include only noise
same_noise = False #Use saved noise seed
get_new_seed = False #Not functioning, keep False
#When same_noise is True, this is used to change seed of random number 
random_seed = 1 # seed 1 is a bad measurment example
# random_seed = 19 # seed 8 is a good measurment example
# random_seed = 89 # seed 89 is unhappy with my SBT


distribution = 1e-3 #Variance of white noise
RTU = True
PMU = False


