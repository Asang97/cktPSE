from pyomo.environ import *
from scripts.Measure import measure_RTU
from scripts.Measure import measure_PMU
import numpy as np
import pandas as pd
from scripts.options import unknown_branch_G, unknown_G_branch_id, unknown_branch_B, unknown_B_branch_id, tolerance, max_iter, flag_tee, model_type, delta_B,get_estB,test_Mc,B_ineq_cosntraint,Obj_scal_factor\
	,w_slack, w_nv, w_nr, w_ni

def test_SBT_on_B(bus, branch, transformer, shunt, slack, rtu):
	nc_objective_value = 4.533e-5

	num_buses = len(bus)
		#number of rtus
	num_rtus = len(rtu)
	
	# Set the model 
	model = ConcreteModel()
	model.name = "SBT_model_SBT_Test"

	model.ipopt_vr_list = Var(range(num_buses))
	model.ipopt_vi_list = Var(range(num_buses))

	model.ipopt_slack_nr = Var()
	model.ipopt_nr_list = Var(range(num_rtus))
	model.ipopt_ni_list = Var(range(num_rtus))
	model.ipopt_nv_list = Var(range(num_rtus))
		
	# define variables
	for bus_ele in bus:
		bus_ele.create_ipopt_bus_vars(model)
		bus_ele.initialize_ipopt_bus_vars()

	for slack_ele in slack:
		slack_ele.create_ipopt_slack_var(model)
		slack_ele.initialize_ipopt_slack_var()

	for rtu_ele in rtu:
		rtu_ele.create_ipopt_noise_vars(model)
		rtu_ele.initialize_ipopt_bus_vars()

	model.consl = ConstraintList()
	
	if unknown_branch_B: # Create and initialize B variables for unknown B 
	
		unknown_B_between = []
		index_unknownB = 0
		
		num_unknownB = len(unknown_B_branch_id)
		model.ipopt_B_list = Var(range(num_unknownB))
		
		# if test_Mc:
		# 	model.ipopt_wr_list = Var(range(num_unknownB))
		# 	model.ipopt_wi_list = Var(range(num_unknownB))
		
		print ("Number of Unknown B:",num_unknownB) # The number of unknown B is equal to the length of unknown B 
		
		branch_to_unknownB_key = {}
		for ele_branch in branch: # TODO Here There will only be 1 Var_B, other will keep the best knwon B
			for ele_unknownB_id in unknown_B_branch_id:
				if ele_branch.id ==  ele_unknownB_id:
					ele_branch.unknown_B = True
					unknown_B_between.append((ele_branch.from_bus, ele_branch.to_bus, ele_branch.id))
					branch_to_unknownB_key.update({ele_branch.id:index_unknownB})
					index_unknownB += 1
			
			if ele_branch.unknown_B: # Here should be if unknown B and if tightening let ele_branch.B map to var_B
				ele_branch.create_ipopt_B_vars(bus,
											branch_to_unknownB_key, 
											model)
				ele_branch.initialize_ipopt_B_vars()
		pass
	
	model.consl.add(expr
						= w_slack*(model.ipopt_slack_nr**2)\
						+ w_nv*sum(rtu_ele.ipopt_nv**2 for rtu_ele in rtu)\
						+ w_nr*sum(rtu_ele.ipopt_nr**2 for rtu_ele in rtu)\
						+ w_ni*sum(rtu_ele.ipopt_ni**2 for rtu_ele in rtu)\
						<=nc_objective_value)
	for bus_ele in bus:
		if bus_ele.flag_ZIbus == 0:
			model.consl.add(expr= 
								sum(ele.calc_real_current_SBT(bus, ele.from_bus) for ele in branch if ele.from_bus == bus_ele.Bus)\
								+ sum(ele.calc_real_current_SBT(bus, ele.to_bus) for ele in branch if ele.to_bus == bus_ele.Bus)\
								+ sum(ele.calc_real_current(bus, ele.from_bus) for ele in transformer if ele.from_bus == bus_ele.Bus)\
								+ sum(ele.calc_real_current(bus, ele.to_bus) for ele in transformer if ele.to_bus == bus_ele.Bus)\
								+ sum(ele.calc_real_current(bus) for ele in shunt if ele.Bus == bus_ele.Bus)\
								# RTU Noise should be indepedent and added sepratly
								# as this is only thing changedby model type
								+ sum(bus_ele.ipopt_vr*rtu_ele.p_mea/((rtu_ele.vmag_mea)**2)\
									- bus_ele.ipopt_vi*rtu_ele.q_mea/((rtu_ele.vmag_mea)**2)\
									+ rtu_ele.ipopt_nr for rtu_ele in rtu if rtu_ele.Bus == bus_ele.Bus)\
									== 0)
						
			model.consl_SBT.add(expr= 
								sum(ele.calc_imag_current_SBT(bus, ele.from_bus) for ele in branch if ele.from_bus == bus_ele.Bus)\
								+ sum(ele.calc_imag_current_SBT(bus, ele.to_bus) for ele in branch if ele.to_bus == bus_ele.Bus)\
								+ sum(ele.calc_imag_current(bus, ele.from_bus) for ele in transformer if ele.from_bus == bus_ele.Bus)\
								+ sum(ele.calc_imag_current(bus, ele.to_bus) for ele in transformer if ele.to_bus == bus_ele.Bus)\
								+ sum(ele.calc_imag_current(bus) for ele in shunt if ele.Bus == bus_ele.Bus)\
								# RTU Noise
								+ sum(bus_ele.ipopt_vi*rtu_ele.p_mea/((rtu_ele.vmag_mea)**2)\
									+ bus_ele.ipopt_vr*rtu_ele.q_mea/((rtu_ele.vmag_mea)**2)\
									+ rtu_ele.ipopt_ni for rtu_ele in rtu if rtu_ele.Bus == bus_ele.Bus)\
									== 0)
		if bus_ele.flag_ZIbus == 1:
			model.consl.add(expr= 
							sum(ele.calc_real_current_SBT(bus, ele.from_bus) for ele in branch if ele.from_bus == bus_ele.Bus)\
						+ sum(ele.calc_real_current_SBT(bus, ele.to_bus) for ele in branch if ele.to_bus == bus_ele.Bus)\
						+ sum(ele.calc_real_current(bus, ele.from_bus) for ele in transformer if ele.from_bus == bus_ele.Bus)\
						+ sum(ele.calc_real_current(bus, ele.to_bus) for ele in transformer if ele.to_bus == bus_ele.Bus)\
						+ sum(ele.calc_real_current(bus) for ele in shunt if ele.Bus == bus_ele.Bus)\
						== 0)
				
			model.consl.add(expr= 
							sum(ele.calc_imag_current_SBT(bus, ele.from_bus) for ele in branch if ele.from_bus == bus_ele.Bus)\
						+ sum(ele.calc_imag_current_SBT(bus, ele.to_bus) for ele in branch if ele.to_bus == bus_ele.Bus)\
						+ sum(ele.calc_imag_current(bus, ele.from_bus) for ele in transformer if ele.from_bus == bus_ele.Bus)\
						+ sum(ele.calc_imag_current(bus, ele.to_bus) for ele in transformer if ele.to_bus == bus_ele.Bus)\
						+ sum(ele.calc_imag_current(bus) for ele in shunt if ele.Bus == bus_ele.Bus)\
						== 0)
	
	model.B_SBT_var = Var()
	
	model.B_lower_obj = Objective(expr= model.B_SBT_var\
							,sense = minimize)
	
	solver = SolverFactory('ipopt')
	solver.options['tol'] = tolerance
	# solver.options["OF_hessian_approximation"]="exact" #This one is not functioning
	solver.options["max_iter"]= max_iter
	# solver.options["print_level"]= 6
	solver.options['print_frequency_iter'] = 1
	solver.options["print_user_options"] = "yes"
	solver.options["print_info_string"] = "yes"
	solver.options["honor_original_bounds"] = "yes"
	solver.options["mu_init"] = 1e-2
	# solver.options["nlp_scaling_method"] = "none"
	# solver.options["nlp_scaling_method"] = "gradient-based"
	solver.options["obj_scaling_factor"] = 1 # Same as setting w for all noise 
	pyomo_options = {"tee": flag_tee}
	solver.solve(model, **pyomo_options)