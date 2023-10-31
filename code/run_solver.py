from scripts.Solve import solve 
from scripts.Solve import multi_solve
import numpy as np
import os

# path to the grid network RAW file
path = os.getcwd()

# case_name="IEEE-118_prior_solution.RAW"
# case_name="case2869pegase_OPF.RAW"
# case_name = "ACTIVSg10k.RAW"
# case_name = "IEEE-14_prior_solution.RAW"
# case_name = 'GS-4_prior_solution.RAW'
# case_name = 'PEGASE-9241_flat_start.RAW'
# case_name = "IEEE57bus.RAW"
case_name = "ACTIVSg200.RAW"

casename = path+"/testcases/"+case_name
# casename = "IPOPT_CASE/code/opf_raw_files/case336.raw"

# the settings for Power flow simulation
settings = {
	"Tolerance": 1E-07,
	"Max Iters": 1000,
	"Limiting":  False,
	"Sparse": True
}

flag_multisolve = False #Turn on if test multiple times
if not flag_multisolve:
	try:
		solve(casename, settings)
	except FileNotFoundError:	
		casename = 'IPOPT_CASE/code/testcases/' + case_name
		solve(casename, settings)
	except:
		print ("Other Error")
