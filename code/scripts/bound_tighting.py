from pyomo.environ import *
from scripts.Measure import measure_RTU
from scripts.Measure import measure_PMU
from models.Buses import Buses
from models.Rtu import Rtu
import numpy as np
import pandas as pd
from scripts.options import unknown_branch_G, unknown_G_branch_id, unknown_branch_B, unknown_B_branch_id, tolerance, max_iter, flag_tee, model_type, delta_B,get_estB,test_Mc,B_ineq_cosntraint,Obj_scal_factor\
	,w_slack, w_nv, w_nr, w_ni
"Note this is a seperate model that only provides mccormick tightend lower and upper bounds"

def SBT_on_B_model(v, Y_final, bus, branch, flag_WGN, transformer, shunt, nc_objective_value, B_l, B_u, slack, rtu):
	#number of buses
	num_buses = len(bus)
	#number of rtus
	num_rtus = len(rtu)
	print ("WHAT IS NLP_NC OBJ VALUE?:",nc_objective_value)
	print(num_buses)
	# Set model to be concrete
	model = ConcreteModel()
	model.name = "SBT_model"
	#define measurement data
	# z_RTU0, z_RTU = measure_RTU(v, bus, branch, transformer, shunt, flag_WGN)

	model.ipopt_vr_list = Var(range(num_buses))
	model.ipopt_vi_list = Var(range(num_buses))

	model.ipopt_slack_nr = Var()
	model.ipopt_nr_list = Var(range(num_rtus))
	model.ipopt_ni_list = Var(range(num_rtus))
	model.ipopt_nv_list = Var(range(num_rtus))

	model.flag_lower = False
	model.flag_upper = False
	model.B_lower_iter = B_l
	model.B_upper_iter = B_u
	
	

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

	#model.n_g = Var(range(1, num_buses+1))
	#model.n_b = Var(range(1, num_buses+1))
	
	# Move this to Slack class 

	
	#Constraint list
	model.consl_SBT = ConstraintList()
	model.consl_Mc	= ConstraintList()
	model.consl = ConstraintList()	

	line_data = {}
	for ele in branch:
		line_data[ele.from_bus, ele.to_bus, ele.id] = {'G': ele.G_l, 'B': ele.B_l, 'b': ele.b}
		line_data[ele.to_bus, ele.from_bus, ele.id] = {'G': ele.G_l, 'B': ele.B_l, 'b': ele.b}
	
	transformer_data = {}
	for ele in transformer:
		transformer_data[ele.from_bus, ele.to_bus] = {'G':ele.G_l, 'B':ele.B_l,'tr':ele.tr, 'ang':ele.ang}
	
	branch_to_unknownB_key = {}
	if unknown_branch_B: # Create and initialize B variables for unknown B 
		
		unknown_B_between = []
		index_unknownB = 0
		
		num_unknownB = len(unknown_B_branch_id)
		model.ipopt_B_list = Var(range(num_unknownB))
		model.ipopt_wr_list = Var(range(num_unknownB))
		model.ipopt_wi_list = Var(range(num_unknownB))
		
		# if test_Mc:
		# 	model.ipopt_wr_list = Var(range(num_unknownB))
		# 	model.ipopt_wi_list = Var(range(num_unknownB))
		
		print ("Number of Unknown B:",num_unknownB) # The number of unknown B is equal to the length of unknown B 
		
		for ele_branch in branch: # TODO Here There will only be 1 Var_B, other will keep the best knwon B
			for ele_unknownB_id in unknown_B_branch_id:
				if ele_branch.id ==  ele_unknownB_id:
					ele_branch.unknown_B = True
					unknown_B_between.append((ele_branch.from_bus, ele_branch.to_bus, ele_branch.id))
					branch_to_unknownB_key.update({ele_branch.id:index_unknownB})
					index_unknownB += 1
			
			if ele_branch.unknown_B: # Here should be if unknown B and if tightening let ele_branch.B map to var_B
				ele_branch.create_ipopt_B_vars(
											branch_to_unknownB_key, 
			 								model)
				ele_branch.initialize_ipopt_B_vars()
	branch_to_unknownG_key = {}

	branch_to_unknownsh_key = {}
	'Not ready to turn on'
	#TODO with more than one unknown B or G, the following code need to be loop

	if unknown_branch_G and not(unknown_branch_B): #Test the unknown G on branch compare (4,3) (3,4) and (1,2)(2,1)
		(a,b) = unknown_G_between
		real_G = [line_data[a,b]["G"]]
		# print("Real G", line_data[a,b]["G"])
		model.G = Var(within = NonNegativeReals)

		# Initialize G
		model.G = 1

		line_data[a, b]['G'] = model.G
		line_data[b, a]['G'] = model.G
		"""
		If true, setting a G at branch (a,b) to be unknown variable G, with
		a initial point set in model.G

		Note if add one unknown G is added to the problem, all two equality
		constraints related to this with G is affected.

		Each G unknown added is replacing 2 affine equlity constraints to
		2 quadratic equlity constraints
		"""
	else:
		pass
	
	
	if unknown_branch_B and not(unknown_branch_G): #Test the unknown B on branch compare (4,3) (3,4) and (1,2)(2,1)
		
		real_B_dict = {}
		num_unknownB = len(unknown_B_branch_id)
		
		# model.ipopt_B_list = Var(range(num_unknownB)) 
		# print ("Number of Unknown B:",num_unknownB) # The number of unknown B is equal to the length of unknown B 
		
		# # Make a num_unknownB to branch_id 
		# index_unknownB = 0
		# for ele_branch in branch:
		# 	for ele_unknownB_id in unknown_B_branch_id:
		# 		if ele_branch.id ==  ele_unknownB_id:
		# 			ele_branch.unknown_B = True
		# 			branch_to_unknownB_key.update({ele_branch.id:index_unknownB})
		# 			index_unknownB += 1
		counter = 0
		real_B_list = []
		for branches in unknown_B_between:
			(a,b,id) = branches # Branch from bus a to bus b			
			
			real_B = [line_data[a,b,id]["B"]]
			real_B_list.append(line_data[a,b,id]["B"])
			
			real_B_dict.update({(a,b,id):real_B})
			real_B_dict.update({(b,a,id):real_B})

			counter += 1
	else:
		pass
	
	if unknown_branch_G and unknown_branch_B:
		pass
	

	
	# Extra constraint that this objective should be smaller than NLP result
	# Objective minB_l minB_u maxB_l and maxB_u will be add in tighten_lower/upper functions 
	model.consl_SBT.add(expr
						= w_slack*(model.ipopt_slack_nr**2)\
						+ w_nv*sum(rtu_ele.ipopt_nv**2 for rtu_ele in rtu)\
						+ w_nr*sum(rtu_ele.ipopt_nr**2 for rtu_ele in rtu)\
						+ w_ni*sum(rtu_ele.ipopt_ni**2 for rtu_ele in rtu)\
						<= nc_objective_value)
	
	# Also add limitation to B
	"range of B"
	model.B_SBT_var = Var()
	
	model.consl_SBT.add(expr= model.B_SBT_var - B_u <= 0) # This is functioning
	model.consl_SBT.add(expr= model.B_SBT_var - B_l >= 0) # This is fuctioning
	

	
	
	############TESTONLY###############
	
	############TESTONLY###############
	# flag_ZIbus = np.zeros(shape=(num_buses+1,))

	# Should not be rtu if ZI 
	
	rtu_bus_list=[]
	all_bus_list=[]
	for rtu_ele in rtu:
		rtu_bus_list.append(rtu_ele.Bus)
	for bus_ele in bus:
		all_bus_list.append(bus_ele.Bus)
	notZI_bus_list = list(set(rtu_bus_list) & set(all_bus_list))
	print("NOT ZI:",notZI_bus_list)

	for ele in notZI_bus_list:
		bus[Buses.all_bus_key_[ele]].flag_ZIbus = 0 # now the default is 1 for this flag
	
	for ele_bus in bus:
		if ele_bus.flag_ZIbus == 1:
			print ("num_ZIbus:",ele_bus.Bus)


	for rtu_ele in rtu:	
		if abs(rtu_ele.p_mea) <= 1e-10 and abs(rtu_ele.q_mea) <= 1e-10:
			bus[Buses.all_bus_key_[rtu_ele.Bus]].flag_ZIbus = 1
			print ("+++++++++++++++++++++++  ZEROINJECTION  ++++++++++++++++++++++++",rtu_ele.Bus) # If this prints, means ZI not fully removed form Rtu list


	# Add Slack Contribution
	for slack_ele in slack:
		rtu_ele = rtu[Rtu.bus_key[slack_ele.Bus]]
		slack_ele.add_ipopt_slack_constraint(model, 
											 rtu_ele.vmag_mea, 
											 bus)
		
	# Define the KCL equations
	for bus_ele in bus:
		
		if B_ineq_cosntraint:
				for branch_ele in branch:
					# if branch_ele.unknown_B:
					if branch_ele.unknown_B and branch_ele.from_bus == bus_ele.Bus: # Wait why this is changing the result?
						# print ("What Branch is this:",branch_ele.from_bus,branch_ele.to_bus)
						branch_ele.add_B_ineq_constraint(model) # For scalling to work this is needed

		for branch_ele in branch:
				unknown_branches = []
				if branch_ele.from_bus == bus_ele.Bus and branch_ele.unknown_B:
					unknown_branches.append(branch_ele)
					branch_ele.create_Mc_vars(branch_to_unknownB_key, branch_to_unknownG_key, branch_to_unknownsh_key, model)
					branch_ele.initialize_Mc_vars(bus)
					branch_ele.initialize_SBT_vars(model)
					branch_ele.Mc_SBT_inequality_constraint(bus,model)
				
		
		if bus_ele.flag_ZIbus == 0:
			# print ("what is the vr at bus",bus_ele.Bus ,"?",value(bus_ele.ipopt_vr ))
			# print ("what is the vi at bus",bus_ele.Bus ,"?",value(bus_ele.ipopt_vi ))
		
		
			#TODO why this constraint?
			# model.consl.add(expr=(model.v_r[i]**2 + model.v_i[i]**2 - z_RTU[i]['v_mag']**2 - model.n_v[i]==0)) # This works
			for rtu_ele in rtu:
				if rtu_ele.Bus == bus_ele.Bus:
					# This constraint is satisfied at the 0 iter
					#TODO Get this dude a convex realxation
					# model.consl.add(expr= (bus_ele.ipopt_vr**2 + bus_ele.ipopt_vi**2 - (rtu_ele.vmag_mea - rtu_ele.ipopt_nv)**2) == 0) # This works 
					model.consl.add(expr= (bus_ele.ipopt_vr**2 + bus_ele.ipopt_vi**2 - (rtu_ele.vmag_mea - rtu_ele.ipopt_nv)**2) <= 0) # This works as conic relaxation of equaliy constraint
					# model.consl.add(expr= (bus_ele.ipopt_vr**2 + bus_ele.ipopt_vi**2 - (rtu_ele.vmag_mea - rtu_ele.ipopt_nv)**2) >= 0)
					pass
			
			"""
			This constraint is needed, with out it we can not squeeze the solution to 2
			IPOPT will find a large number of solution near the initial points
			"""
			"""
			for ele in slack:
			ele.add_slack_constraints(model)
			"""
		
			#Noise -> model.n_r -> G_noise 
						# Model 1 = (G + jB)(Vr + jVi) + Ir_noise + jIi_noise
						# Model 2 = (P + P_noise + jQ + jQ_noise)/(Vr - jVi)
						# Model 3 = (G + G_noise + jB + jBnoise)(Vr + jVi)
						# (G + G_noise)Vr - (B + B_noise)Vi = G.Vr - BVi + G_noise.Vr - B_noise.Vi
						# Ir = GVr - BVi
			
			

			if model_type == 4:

				# For Non_convex
				# All KCL seems not functioning
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
				
				model.consl.add(expr= 
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
				
				# model.consl_Mc.add(expr= 
				# 			sum(ele.calc_real_current_Mc(bus, ele.from_bus) for ele in branch if ele.from_bus == bus_ele.Bus and ele.unknown_B)\
				# 			+ sum(ele.calc_real_current_Mc(bus, ele.to_bus) for ele in branch if ele.to_bus == bus_ele.Bus and ele.unknown_B)\
				# 			+ sum(ele.calc_real_current(bus, ele.from_bus) for ele in branch if ele.from_bus == bus_ele.Bus and not ele.unknown_B)\
				# 			+ sum(ele.calc_real_current(bus, ele.to_bus) for ele in branch if ele.to_bus == bus_ele.Bus and not ele.unknown_B)\
				# 			+ sum(ele.calc_real_current(bus, ele.from_bus) for ele in transformer if ele.from_bus == bus_ele.Bus)\
				# 			+ sum(ele.calc_real_current(bus, ele.to_bus) for ele in transformer if ele.to_bus == bus_ele.Bus)\
				# 			+ sum(ele.calc_real_current(bus) for ele in shunt if ele.Bus == bus_ele.Bus)\
				# 			# RTU Noise
				# 			+ sum(bus_ele.ipopt_vr*rtu_ele.p_mea/((rtu_ele.vmag_mea)**2)\
				# 				- bus_ele.ipopt_vi*rtu_ele.q_mea/((rtu_ele.vmag_mea)**2)\
				# 				+ rtu_ele.ipopt_nr for rtu_ele in rtu if rtu_ele.Bus == bus_ele.Bus)\
				# 				== 0)
					
				# model.consl_Mc.add(expr= 
				# 			sum(ele.calc_imag_current_Mc(bus, ele.from_bus) for ele in branch if ele.from_bus == bus_ele.Bus and ele.unknown_B)\
				# 			+ sum(ele.calc_imag_current_Mc(bus, ele.to_bus) for ele in branch if ele.to_bus == bus_ele.Bus and ele.unknown_B)\
				# 			+ sum(ele.calc_imag_current(bus, ele.from_bus) for ele in branch if ele.from_bus == bus_ele.Bus and not ele.unknown_B)\
				# 			+ sum(ele.calc_imag_current(bus, ele.to_bus) for ele in branch if ele.to_bus == bus_ele.Bus and not ele.unknown_B)\
				# 			+ sum(ele.calc_imag_current(bus, ele.from_bus) for ele in transformer if ele.from_bus == bus_ele.Bus)\
				# 			+ sum(ele.calc_imag_current(bus, ele.to_bus) for ele in transformer if ele.to_bus == bus_ele.Bus)\
				# 			+ sum(ele.calc_imag_current(bus) for ele in shunt if ele.Bus == bus_ele.Bus)\
				# 			# RTU Noise
				# 			+ sum(bus_ele.ipopt_vi*rtu_ele.p_mea/((rtu_ele.vmag_mea)**2)\
				# 				+ bus_ele.ipopt_vr*rtu_ele.q_mea/((rtu_ele.vmag_mea)**2)\
				# 				+ rtu_ele.ipopt_ni for rtu_ele in rtu if rtu_ele.Bus == bus_ele.Bus)\
				# 				== 0)
			
				# print ("What is Real residual at bus",bus_ele.Bus,":",(
				# 	value(sum(ele.calc_real_current(bus, ele.from_bus) for ele in branch if ele.from_bus == bus_ele.Bus)\
				# 		+ sum(ele.calc_real_current(bus, ele.to_bus) for ele in branch if ele.to_bus == bus_ele.Bus)\
				# 		+ sum(ele.calc_real_current(bus, ele.from_bus) for ele in transformer if ele.from_bus == bus_ele.Bus)\
				# 		+ sum(ele.calc_real_current(bus, ele.to_bus) for ele in transformer if ele.to_bus == bus_ele.Bus)\
				# 		+ sum(ele.calc_real_current(bus) for ele in shunt if ele.Bus == bus_ele.Bus)),\
				# 		# RTU Noise
				# 		value(+ sum(bus_ele.ipopt_vr*rtu_ele.p_mea/((rtu_ele.vmag_mea)**2)\
				# 		      - bus_ele.ipopt_vi*rtu_ele.q_mea/((rtu_ele.vmag_mea)**2)\
				# 		      + rtu_ele.ipopt_nr for rtu_ele in rtu if rtu_ele.Bus == bus_ele.Bus)\
				# 		    	)))
				
				# print ("What is Imaginary residual bus",bus_ele.Bus,":",(
				# 	value(sum(ele.calc_imag_current(bus, ele.from_bus) for ele in branch if ele.from_bus == bus_ele.Bus)\
				# 		+ sum(ele.calc_imag_current(bus, ele.to_bus) for ele in branch if ele.to_bus == bus_ele.Bus)\
				# 		+ sum(ele.calc_imag_current(bus, ele.from_bus) for ele in transformer if ele.from_bus == bus_ele.Bus)\
				# 		+ sum(ele.calc_imag_current(bus, ele.to_bus) for ele in transformer if ele.to_bus == bus_ele.Bus)\
				# 		+ sum(ele.calc_imag_current(bus) for ele in shunt if ele.Bus == bus_ele.Bus)),\
				# 		# RTU Noise
				# 		value(+ sum(bus_ele.ipopt_vi*rtu_ele.p_mea/((rtu_ele.vmag_mea)**2)\
				# 			  + bus_ele.ipopt_vr*rtu_ele.q_mea/((rtu_ele.vmag_mea)**2)\
				# 		      + rtu_ele.ipopt_ni for rtu_ele in rtu if rtu_ele.Bus == bus_ele.Bus)\
				# 		    	)))
			
			if model_type == 3:
				model.consl.add(expr=
						sum(ele.calc_real_current(model.v_r[ele.from_bus], model.v_i[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.to_bus], i) for ele in branch if ele.from_bus == i)\
						+ sum(ele.calc_real_current(model.v_r[ele.from_bus], model.v_i[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.to_bus], i) for ele in branch if ele.to_bus == i)\
						# Up there should be branches' current
						+ sum(ele.calc_real_current(model.v_r[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.from_bus], model.v_i[ele.to_bus], i) for ele in transformer if ele.from_bus == i)\
						+ sum(ele.calc_real_current(model.v_r[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.from_bus], model.v_i[ele.to_bus], i) for ele in transformer if ele.to_bus == i)\
						+ sum(ele.calc_real_current(model.v_r[ele.Bus], model.v_i[ele.Bus]) for ele in shunt if ele.Bus == i)\
						#Up there should add up to all current flow at node i except I_RTU
						+ model.v_r[i]*z_RTU[i]['p']/((z_RTU[i]['v_mag'])**2)\
						- model.v_i[i]*z_RTU[i]['q']/((z_RTU[i]['v_mag'])**2)\
						#This one should be I_RTU
						+ model.v_r[i]*model.n_r[i]\
						- model.v_i[i]*model.n_i[i] == 0)
						#And the noise term
		
				model.consl.add(expr= 
						sum(ele.calc_imag_current(model.v_r[ele.from_bus], model.v_i[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.to_bus], i) for ele in branch if ele.from_bus == i)\
						+ sum(ele.calc_imag_current(model.v_r[ele.from_bus], model.v_i[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.to_bus], i) for ele in branch if ele.to_bus == i)\
						+ sum(ele.calc_imag_current(model.v_r[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.from_bus], model.v_i[ele.to_bus], i) for ele in transformer if ele.from_bus == i)\
						+ sum(ele.calc_imag_current(model.v_r[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.from_bus], model.v_i[ele.to_bus], i) for ele in transformer if ele.to_bus == i)\
						+ sum(ele.calc_imag_current(model.v_r[ele.Bus], model.v_i[ele.Bus], i) for ele in shunt if ele.Bus == i)\
						+ model.v_i[i]*z_RTU[i]['p']/((z_RTU[i]['v_mag'])**2)\
						+ model.v_r[i]*z_RTU[i]['q']/((z_RTU[i]['v_mag'])**2)\
						+ model.v_i[i]*model.n_r[i]\
						+ model.v_r[i]*model.n_i[i] == 0)
				
			
			elif model_type == 1:
				model.consl.add(expr= 
						sum(ele.calc_real_current(model.v_r[ele.from_bus], model.v_i[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.to_bus], i) for ele in branch if ele.from_bus == i)\
						+ sum(ele.calc_real_current(model.v_r[ele.from_bus], model.v_i[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.to_bus], i) for ele in branch if ele.to_bus == i)\
						+ sum(ele.calc_real_current(model.v_r[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.from_bus], model.v_i[ele.to_bus], i) for ele in transformer if ele.from_bus == i)\
						+ sum(ele.calc_real_current(model.v_r[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.from_bus], model.v_i[ele.to_bus], i) for ele in transformer if ele.to_bus == i)\
						+ sum(ele.calc_real_current(model.v_r[ele.Bus], model.v_i[ele.Bus]) for ele in shunt if ele.Bus == i)\
						+ model.v_r[i]*z_RTU[i]['p']/((z_RTU[i]['v_mag'])**2)\
						- model.v_i[i]*z_RTU[i]['q']/((z_RTU[i]['v_mag'])**2)\
						+ model.n_r[i] == 0)
		
				model.consl.add(expr= 
						sum(ele.calc_imag_current(model.v_r[ele.from_bus], model.v_i[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.to_bus], i) for ele in branch if ele.from_bus == i)\
						+ sum(ele.calc_imag_current(model.v_r[ele.from_bus], model.v_i[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.to_bus], i) for ele in branch if ele.to_bus == i)\
						+ sum(ele.calc_imag_current(model.v_r[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.from_bus], model.v_i[ele.to_bus], i) for ele in transformer if ele.from_bus == i)\
						+ sum(ele.calc_imag_current(model.v_r[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.from_bus], model.v_i[ele.to_bus], i) for ele in transformer if ele.to_bus == i)\
						+ sum(ele.calc_imag_current(model.v_r[ele.Bus], model.v_i[ele.Bus]) for ele in shunt if ele.Bus == i)\
						+ model.v_i[i]*z_RTU[i]['p']/((z_RTU[i]['v_mag'])**2)\
						+ model.v_r[i]*z_RTU[i]['q']/((z_RTU[i]['v_mag'])**2)\
						+ model.n_i[i] == 0)
			
			elif model_type == 2:
				model.consl.add(expr= 
						sum(ele.calc_real_current(model.v_r[ele.from_bus], model.v_i[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.to_bus], i) for ele in branch if ele.from_bus == i)\
						+ sum(ele.calc_real_current(model.v_r[ele.from_bus], model.v_i[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.to_bus], i) for ele in branch if ele.to_bus == i)\
						+ sum(ele.calc_real_current(model.v_r[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.from_bus], model.v_i[ele.to_bus], i) for ele in transformer if ele.from_bus == i)\
						+ sum(ele.calc_real_current(model.v_r[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.from_bus], model.v_i[ele.to_bus], i) for ele in transformer if ele.to_bus == i)\
						+ sum(ele.calc_real_current(model.v_r[ele.Bus], model.v_i[ele.Bus]) for ele in shunt if ele.Bus == i)\
						+ (model.v_r[i]*(z_RTU[i]['p'] + model.n_r[i])\
						-  model.v_i[i]*(z_RTU[i]['q'] + model.n_i[i]))\
						/  (model.v_r[i]**2 + model.v_i[i]**2)\
						== 0)
					
		
				model.consl.add(expr= 
						sum(ele.calc_imag_current(model.v_r[ele.from_bus], model.v_i[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.to_bus], i) for ele in branch if ele.from_bus == i)\
						+ sum(ele.calc_imag_current(model.v_r[ele.from_bus], model.v_i[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.to_bus], i) for ele in branch if ele.to_bus == i)\
						+ sum(ele.calc_imag_current(model.v_r[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.from_bus], model.v_i[ele.to_bus], i) for ele in transformer if ele.from_bus == i)\
						+ sum(ele.calc_imag_current(model.v_r[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.from_bus], model.v_i[ele.to_bus], i) for ele in transformer if ele.to_bus == i)\
						+ sum(ele.calc_imag_current(model.v_r[ele.Bus], model.v_i[ele.Bus]) for ele in shunt if ele.Bus == i)\
						+ (model.v_i[i]*(z_RTU[i]['p'] + model.n_r[i])\
						+  model.v_r[i]*(z_RTU[i]['q'] + model.n_i[i]))\
						/  (model.v_r[i]**2 + model.v_i[i]**2)\
						== 0)
		
		elif bus_ele.flag_ZIbus == 1:
			
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
			
			# TODO
			# model.consl.add(expr= model.v_r[i] >= 0)
			# model.consl.add(expr= model.v_i[i] <= 0)
			# model.consl.add(expr= sum(Transformers.calc_real_current(v_r[i], v_r[j],v_i[i], v_i[j], bus) for j in range (1, num_buses+1) if (i,j) in transformer_data.keys()) == 0)
	
	# Define objectives of SBT
	model.B_lower_obj = Objective(expr= model.B_SBT_var\
							,sense = minimize)
	model.B_upper_obj =Objective(expr= model.B_SBT_var\
							,sense = maximize)
	
	model.noise = Objective(expr
			 					= w_slack*(model.ipopt_slack_nr**2)\
			 					+ w_nv*sum(rtu_ele.ipopt_nv**2 for rtu_ele in rtu)\
			 					+ w_nr*sum(rtu_ele.ipopt_nr**2 for rtu_ele in rtu)\
		  						+ w_ni*sum(rtu_ele.ipopt_ni**2 for rtu_ele in rtu)\
							,sense = minimize)
	
	return model
	
def tighten_lower(model):
	
	"model.B is var, model.B_lower_obj is objective, model.B_lower_iter is constant for iteration"
	
	# model.del_component(Objective) 
	model.B_lower_obj.activate()
	model.B_upper_obj.deactivate()
	model.noise.deactivate()
	model.consl_SBT.activate()
	model.consl_Mc.deactivate()
	model.B_SBT_var.value =  model.B_lower_iter
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
	solver.options["obj_scaling_factor"] = 1e-9 # Same as setting w for all noise 
	pyomo_options = {"tee": flag_tee}
	solver.solve(model, **pyomo_options)
	
	"Check flag"
	print ("B_lower tighten form", value(model.B_lower_iter), "to", value(model.B_SBT_var))
	B_lower_new =  value(model.B_SBT_var)
	if ((B_lower_new - model.B_lower_iter)/model.B_lower_iter)**2 <= 1e-9:
		"send flag to model if true"
		model.flag_lower = True
	"update the model"
	model.B_lower_iter = B_lower_new
	model.consl.add(expr= model.B_SBT_var - B_lower_new >= 0)
	print ("B_lower", B_lower_new)
	return B_lower_new

def tighten_upper(model):
	"model.B is var, model.B_lower_obj is objective, model.B_lower_iter is constant for iteration"
	
	# model.del_component(Objective)
	model.B_lower_obj.deactivate()
	model.B_upper_obj.activate()
	model.noise.deactivate()
	model.consl_SBT.activate()
	model.consl_Mc.deactivate()
	model.B_SBT_var.value =  model.B_upper_iter
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

	"Check flag"
	print ("B_upper tighten form", value(model.B_upper_iter), "to", value(model.B_SBT_var))
	B_upper_new =  value(model.B_SBT_var)
	if ((B_upper_new - model.B_upper_iter)/model.B_upper_iter)**2 <= 1e-9:
		"send flag to model if true"
		model.flag_upper = True
	"update the model"
	model.B_upper_iter = B_upper_new
	model.consl.add(expr= model.B_SBT_var - B_upper_new <= 0)
	print ("B_upper",B_upper_new)
	return B_upper_new

def tol_check(model):
	if model.flag_upper and model.flag_lower:
		return True
	else:
		return False

def Test_run(model):
	
	model.B_lower_obj.deactivate()
	model.B_upper_obj.deactivate()
	model.noise.activate()
	
	
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
	print (model.name)
	result = solver.solve(model, **pyomo_options)
	print(result)