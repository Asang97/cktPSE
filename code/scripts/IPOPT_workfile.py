from pyomo.environ import Var, ConcreteModel, ConstraintList, NonNegativeReals, Objective, minimize, SolverFactory, value, expr
from scripts.Measure import measure_RTU
from scripts.Measure import measure_PMU
from models.Buses import Buses
from models.Rtu import Rtu
import numpy as np
import pandas as pd
from scripts.options import unknown_branch_G, unknown_G_branch_id, unknown_branch_B, unknown_B_branch_id, \
	unknown_branch_sh, unknown_sh_branch_id, unknown_transformer_tr, unknown_tr_transformer_id, \
		tolerance, max_iter, flag_tee, model_type, delta_B,get_estB,test_Mc,B_ineq_cosntraint,Obj_scal_factor\
	,w_slack, w_nv, w_nr, w_ni



def State_estimation(v, Y_final, bus, branch, flag_WGN, transformer, shunt, slack, rtu):
	#number of buses
	num_buses = len(bus)
	#number of rtus
	num_rtus = len(rtu)

	print(num_buses)
	# Set model to be concrete
	model = ConcreteModel()
	model.name = "NLP_NC"
	#define measurement data
	# z_RTU0, z_RTU = measure_RTU(v, bus, branch, transformer, shunt, flag_WGN)

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

	#model.n_g = Var(range(1, num_buses+1))
	#model.n_b = Var(range(1, num_buses+1))
	
	# Move this to Slack class 

	
	#Constraint list
	model.consl = ConstraintList()	

	line_data = {}
	for ele in branch:
		line_data[ele.from_bus, ele.to_bus, ele.id] = {'G': ele.G_l, 'B': ele.B_l, 'b': ele.b}
		line_data[ele.to_bus, ele.from_bus, ele.id] = {'G': ele.G_l, 'B': ele.B_l, 'b': ele.b}
	
	transformer_data = {}
	for ele in transformer:
		transformer_data[ele.from_bus, ele.to_bus] = {'G':ele.G_l, 'B':ele.B_l,'tr':ele.tr, 'ang':ele.ang}
	
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
		for ele_branch in branch:
			for ele_unknownB_id in unknown_B_branch_id:
				if ele_branch.id ==  ele_unknownB_id:
					ele_branch.unknown_B = True
					unknown_B_between.append((ele_branch.from_bus, ele_branch.to_bus, ele_branch.id))
					branch_to_unknownB_key.update({ele_branch.id:index_unknownB})
					index_unknownB += 1
			
			if ele_branch.unknown_B:
				ele_branch.create_ipopt_B_vars(
											branch_to_unknownB_key, 
			 								model)
				ele_branch.initialize_ipopt_B_vars()
		
		real_B_dict = {}
		counter = 0
		real_B_list = []
		
		for branches in unknown_B_between:
			(a,b,id) = branches # Branch from bus a to bus b
			
			# TODO also Move this to branch class
			# Change the data in Class branch					
			
			real_B = [line_data[a,b,id]["B"]]
			real_B_list.append(line_data[a,b,id]["B"])
			
			real_B_dict.update({(a,b,id):real_B})
			real_B_dict.update({(b,a,id):real_B})

			counter += 1
	
	if unknown_branch_G: # Create and initialize G variables for unknown G 
		
		unknown_G_between = []
		index_unknownG = 0
		
		num_unknownG = len(unknown_G_branch_id)
		print ("num_unknownG",num_unknownG)
		model.ipopt_G_list = Var(range(num_unknownG))
		
		# if test_Mc:
		# 	model.ipopt_wr_list = Var(range(num_unknownB))
		# 	model.ipopt_wi_list = Var(range(num_unknownB))
		
		print ("Number of Unknown G:",num_unknownG) # The number of unknown G is equal to the length of unknown G 
		
		branch_to_unknownG_key = {}
		for ele_branch in branch:
			for ele_unknownG_id in unknown_G_branch_id:
				if ele_branch.id ==  ele_unknownG_id:
					ele_branch.unknown_G = True
					unknown_G_between.append((ele_branch.from_bus, ele_branch.to_bus, ele_branch.id))
					branch_to_unknownG_key.update({ele_branch.id:index_unknownG})
					index_unknownG += 1
			
			if ele_branch.unknown_G:
				ele_branch.create_ipopt_G_vars(
											branch_to_unknownG_key, 
			 								model)
				ele_branch.initialize_ipopt_G_vars()

	if unknown_branch_sh: # Create and initialize sh variables for unknown shunts
		
		unknown_sh_between = []
		index_unknownsh = 0
		
		num_unknownsh = len(unknown_sh_branch_id)
		model.ipopt_sh_list = Var(range(num_unknownsh))
		
		# if test_Mc:
		# 	model.ipopt_wr_list = Var(range(num_unknownB))
		# 	model.ipopt_wi_list = Var(range(num_unknownB))
		
		print ("Number of Unknown sh:",num_unknownsh) # The number of unknown B is equal to the length of unknown B 
		
		branch_to_unknownsh_key = {}
		for ele_branch in branch:
			for ele_unknownsh_id in unknown_sh_branch_id:
				if ele_branch.id ==  ele_unknownsh_id:
					ele_branch.unknown_sh = True
					unknown_sh_between.append((ele_branch.from_bus, ele_branch.to_bus, ele_branch.id))
					branch_to_unknownsh_key.update({ele_branch.id:index_unknownsh})
					index_unknownsh += 1
			
			if ele_branch.unknown_sh:
				ele_branch.create_ipopt_sh_vars(
											branch_to_unknownsh_key, 
			 								model)
				ele_branch.initialize_ipopt_sh_vars()

		if unknown_transformer_tr: # Create and initialize B variables for unknown B 
		
			unknown_tr_between = []
			index_unknowntr = 0
			
			num_unknowntr = len(unknown_tr_transformer_id)
			model.ipopt_tr_list = Var(range(num_unknowntr))
			
			# if test_Mc:
			# 	model.ipopt_wr_list = Var(range(num_unknownB))
			# 	model.ipopt_wi_list = Var(range(num_unknownB))
			
			print ("Number of Unknown tr:",num_unknowntr) # The number of unknown B is equal to the length of unknown B 
			
			xfm_to_unknowntr_key = {}
			for ele_xfm in transformer:
				
				for ele_unknowntr_id in unknown_tr_transformer_id:
					if ele_xfm.id ==  ele_unknowntr_id:
						ele_xfm.unknown_tr = True
						unknown_tr_between.append((ele_xfm.from_bus, ele_xfm.to_bus, ele_xfm.id))
						xfm_to_unknowntr_key.update({ele_xfm.id:index_unknowntr})
						index_unknowntr += 1
				
				if ele_xfm.unknown_tr:
					ele_xfm.create_ipopt_tr_vars(
												xfm_to_unknowntr_key, 
												model)
					ele_xfm.initialize_ipopt_tr_vars()

	'Not ready to turn on'
	#TODO with more than one unknown B or G, the following code need to be loop

	# if unknown_branch_G: #Test the unknown G on branch compare (4,3) (3,4) and (1,2)(2,1)
	# 	(a,b) = unknown_G_between
	# 	real_G = [line_data[a,b]["G"]]
	# 	# print("Real G", line_data[a,b]["G"])
	# 	model.G = Var(within = NonNegativeReals)

	# 	# Initialize G
	# 	model.G = 1

	# 	line_data[a, b]['G'] = model.G
	# 	line_data[b, a]['G'] = model.G
	# 	"""
	# 	If true, setting a G at branch (a,b) to be unknown variable G, with
	# 	a initial point set in model.G

	# 	Note if add one unknown G is added to the problem, all two equality
	# 	constraints related to this with G is affected.

	# 	Each G unknown added is replacing 2 affine equlity constraints to
	# 	2 quadratic equlity constraints
	# 	"""
	# else:
	# 	pass
	
	
	# if unknown_branch_B: #Test the unknown B on branch compare (4,3) (3,4) and (1,2)(2,1)
		
	# 	real_B_dict = {}
	# 	num_unknownB = len(unknown_B_branch_id)
		
	# 	# model.ipopt_B_list = Var(range(num_unknownB)) 
	# 	# print ("Number of Unknown B:",num_unknownB) # The number of unknown B is equal to the length of unknown B 
		
	# 	# # Make a num_unknownB to branch_id 
	# 	# index_unknownB = 0
	# 	# for ele_branch in branch:
	# 	# 	for ele_unknownB_id in unknown_B_branch_id:
	# 	# 		if ele_branch.id ==  ele_unknownB_id:
	# 	# 			ele_branch.unknown_B = True
	# 	# 			branch_to_unknownB_key.update({ele_branch.id:index_unknownB})
	# 	# 			index_unknownB += 1
	# 	counter = 0
	# 	real_B_list = []
	# 	for branches in unknown_B_between:
	# 		(a,b,id) = branches # Branch from bus a to bus b
			
	# 		# TODO also Move this to branch class
	# 		# Change the data in Class branch					
			
	# 		real_B = [line_data[a,b,id]["B"]]
	# 		real_B_list.append(line_data[a,b,id]["B"])
			
	# 		real_B_dict.update({(a,b,id):real_B})
	# 		real_B_dict.update({(b,a,id):real_B})

	# 		counter += 1
	# else:
	# 	pass
	
	
	# Define the objective function to minimize noise" 
	model.noise = Objective(expr
			 					= w_slack*(model.ipopt_slack_nr**2)\
			 					+ w_nv*sum(rtu_ele.ipopt_nv**2 for rtu_ele in rtu)\
			 					+ w_nr*sum(rtu_ele.ipopt_nr**2 for rtu_ele in rtu)\
		  						+ w_ni*sum(rtu_ele.ipopt_ni**2 for rtu_ele in rtu)\
							,sense = minimize)
	

	
	
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
	# print("NOT ZI:",notZI_bus_list)

	for ele in notZI_bus_list:
		bus[Buses.all_bus_key_[ele]].flag_ZIbus = 0 # now the default is 1 for this flag
	
	count_ZI = 0
	for ele_bus in bus:
		if ele_bus.flag_ZIbus == 1:
			count_ZI  +=  1
			# print ("num_ZIbus:",ele_bus.Bus) # TO show whitch buses are ZI
		
			


	for rtu_ele in rtu:	
		if abs(rtu_ele.p_mea) <= 1e-6 and abs(rtu_ele.q_mea) <= 1e-6:
			bus[Buses.all_bus_key_[rtu_ele.Bus]].flag_ZIbus = 1
			print ("+++++++++++++++++++++++  ALEART: ZEROINJECTION UNREMOVED  ++++++++++++++++++++++++",rtu_ele.Bus) # If this prints, means ZI not fully removed form Rtu list


	# Add Slack Contribution
	for slack_ele in slack:
		rtu_ele = rtu[Rtu.bus_key[slack_ele.Bus]]
		slack_ele.add_ipopt_slack_constraint(model, 
					                         rtu_ele.vmag_mea, 
											 bus)
		
		#TODO: WHAT ABOUT SLACK RTU G and B constraint

	# Define the KCL equations
	for bus_ele in bus:
		
		if B_ineq_cosntraint:
				for branch_ele in branch:
					# if branch_ele.unknown_B:
					if branch_ele.unknown_B and branch_ele.from_bus == bus_ele.Bus: # Wait why this is changing the result?
						# print ("What Branch is this:",branch_ele.from_bus,branch_ele.to_bus)
						branch_ele.add_B_ineq_constraint(model) # For scalling to work this is needed
				
		
		if bus_ele.flag_ZIbus == 0:
			# print ("what is the vr at bus",bus_ele.Bus ,"?",value(bus_ele.ipopt_vr ))
			# print ("what is the vi at bus",bus_ele.Bus ,"?",value(bus_ele.ipopt_vi ))
		
		
			#TODO why this constraint?
			# model.consl.add(expr=(model.v_r[i]**2 + model.v_i[i]**2 - z_RTU[i]['v_mag']**2 - model.n_v[i]==0)) # This works
			for rtu_ele in rtu:
				if rtu_ele.Bus == bus_ele.Bus:
					# This constraint is satisfied at the 0 iter
					# model.consl.add(expr= (bus_ele.ipopt_vr**2 + bus_ele.ipopt_vi**2 - (rtu_ele.vmag_mea - rtu_ele.ipopt_nv)**2) == 0) # This works 
					pass
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
			
			if model_type == 666: # Only to test How it will work with no constraints
				pass

			if model_type == 4:

				# For Non_convex
			
				model.consl.add(expr= 
						sum(ele.calc_real_current(bus, ele.from_bus) for ele in branch if ele.from_bus == bus_ele.Bus)\
						+ sum(ele.calc_real_current(bus, ele.to_bus) for ele in branch if ele.to_bus == bus_ele.Bus)\
						+ sum(ele.calc_real_current(bus, ele.from_bus) for ele in transformer if ele.from_bus == bus_ele.Bus)\
						+ sum(ele.calc_real_current(bus, ele.to_bus) for ele in transformer if ele.to_bus == bus_ele.Bus)\
						+ sum(ele.calc_real_current(bus) for ele in shunt if ele.Bus == bus_ele.Bus)\
						# RTU Noise
						+ sum(bus_ele.ipopt_vr*rtu_ele.p_mea/((rtu_ele.vmag_mea)**2)\
							- bus_ele.ipopt_vi*rtu_ele.q_mea/((rtu_ele.vmag_mea)**2)\
							+ rtu_ele.ipopt_nr for rtu_ele in rtu if rtu_ele.Bus == bus_ele.Bus)\
							== 0)
				
				model.consl.add(expr= 
						sum(ele.calc_imag_current(bus, ele.from_bus) for ele in branch if ele.from_bus == bus_ele.Bus)\
						+ sum(ele.calc_imag_current(bus, ele.to_bus) for ele in branch if ele.to_bus == bus_ele.Bus)\
						+ sum(ele.calc_imag_current(bus, ele.from_bus) for ele in transformer if ele.from_bus == bus_ele.Bus)\
						+ sum(ele.calc_imag_current(bus, ele.to_bus) for ele in transformer if ele.to_bus == bus_ele.Bus)\
						+ sum(ele.calc_imag_current(bus) for ele in shunt if ele.Bus == bus_ele.Bus)\
						# RTU Noise
						+ sum(bus_ele.ipopt_vi*rtu_ele.p_mea/((rtu_ele.vmag_mea)**2)\
							+ bus_ele.ipopt_vr*rtu_ele.q_mea/((rtu_ele.vmag_mea)**2)\
							+ rtu_ele.ipopt_ni for rtu_ele in rtu if rtu_ele.Bus == bus_ele.Bus)\
							== 0)
			
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
							sum(ele.calc_real_current(bus, ele.from_bus) for ele in branch if ele.from_bus == bus_ele.Bus)\
						+ sum(ele.calc_real_current(bus, ele.to_bus) for ele in branch if ele.to_bus == bus_ele.Bus)\
						+ sum(ele.calc_real_current(bus, ele.from_bus) for ele in transformer if ele.from_bus == bus_ele.Bus)\
						+ sum(ele.calc_real_current(bus, ele.to_bus) for ele in transformer if ele.to_bus == bus_ele.Bus)\
						+ sum(ele.calc_real_current(bus) for ele in shunt if ele.Bus == bus_ele.Bus)\
						== 0)
				
			model.consl.add(expr= 
							sum(ele.calc_imag_current(bus, ele.from_bus) for ele in branch if ele.from_bus == bus_ele.Bus)\
						+ sum(ele.calc_imag_current(bus, ele.to_bus) for ele in branch if ele.to_bus == bus_ele.Bus)\
						+ sum(ele.calc_imag_current(bus, ele.from_bus) for ele in transformer if ele.from_bus == bus_ele.Bus)\
						+ sum(ele.calc_imag_current(bus, ele.to_bus) for ele in transformer if ele.to_bus == bus_ele.Bus)\
						+ sum(ele.calc_imag_current(bus) for ele in shunt if ele.Bus == bus_ele.Bus)\
						== 0)
				
			# model.consl.add(expr= 
			# 	  sum(ele.calc_real_current(bus, i) for ele in branch if ele.from_bus == i)\
			# 	+ sum(ele.calc_real_current(bus, i) for ele in branch if ele.to_bus == i)\
			# 	+ sum(ele.calc_real_current(bus, i) for ele in transformer if ele.from_bus == i)\
			# 	+ sum(ele.calc_real_current(bus, i) for ele in transformer if ele.to_bus == i)\
			# 	+ sum(ele.calc_real_current(bus) for ele in shunt if ele.Bus == i)\
			# 	== 0)
				
			# model.consl.add(expr= 
			# 	  sum(ele.calc_imag_current(bus, i) for ele in branch if ele.from_bus == i)\
			# 	+ sum(ele.calc_imag_current(bus, i) for ele in branch if ele.to_bus == i)\
			# 	+ sum(ele.calc_imag_current(bus, i) for ele in transformer if ele.from_bus == i)\
			# 	+ sum(ele.calc_imag_current(bus, i) for ele in transformer if ele.to_bus == i)\
			# 	+ sum(ele.calc_imag_current(bus) for ele in shunt if ele.Bus == i)\
			# 	== 0)
		
			# For Zero Injection bus, all currents add up to zero
	
	# Fixing B[0]
	# model.ipopt_B_list[0].fix(-245.185584)

	print ("The number of ZI is", count_ZI)

	if True:
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
		solver.options["obj_scaling_factor"] = Obj_scal_factor # Same as setting w for all noise 
		pyomo_options = {"tee": flag_tee}
	
	else:
		solver = SolverFactory('multistart')
	
	# Add inequality constraints here
	# model.consl.add(expr=model.G>=0)
	result = solver.solve(model, **pyomo_options)
	# print(result)
	model.solutions.store_to(result)
	#result.write()
	
	# print(sum(ele.calc_real_current(bus, ele.from_bus) for ele in branch if ele.from_bus == 68 and ele.to_bus == 116))
	# print(value(sum(ele.calc_real_current(bus, ele.from_bus) for ele in branch if ele.from_bus == 68 and ele.to_bus == 116)))
	# print(sum(ele.calc_imag_current(bus, ele.from_bus) for ele in branch if ele.from_bus == 68 and ele.to_bus == 116))
	# print(value(sum(ele.calc_imag_current(bus, ele.from_bus) for ele in branch if ele.from_bus == 68 and ele.to_bus == 116)))
	# print(value(sum(ele_rtu.ipopt_nr for ele_rtu in rtu if ele_rtu.Bus == 116)))
	# print(value(sum(ele_rtu.ipopt_ni for ele_rtu in rtu if ele_rtu.Bus == 116)))
	# print(value(sum(ele_rtu.ipopt_nv for ele_rtu in rtu if ele_rtu.Bus == 116)))
	
	# Access the solver status and solution
	# print(result)
	# print(result.solver.status)
	# print(result.solver.termination_condition)

	est_vr= [value(bus_ele.ipopt_vr) for bus_ele in bus]
	est_vi= [value(bus_ele.ipopt_vi) for bus_ele in bus]
	print (np.size(est_vr))
	#TODO first get the correct optiaml objective
	optimal_values_Ir = [sum(value(model.ipopt_nr_list[key])**2 for key in model.ipopt_nr_list)]
	optimal_values_Ii = [sum(value(model.ipopt_ni_list[key])**2 for key in model.ipopt_ni_list)]
	optimal_values_vmag = [sum(value(model.ipopt_nv_list[key])**2 for key in model.ipopt_nv_list)]
	optimal_values_vslack = [sum(value(model.ipopt_slack_nr[key])**2 for key in model.ipopt_slack_nr)]
	
	optimal_values =optimal_values_vslack + optimal_values_vmag + optimal_values_Ir + optimal_values_Ii
	optimal_values_sum = optimal_values_Ir[0] + optimal_values_Ii[0] + optimal_values_vmag[0] + optimal_values_vslack[0]
	
	print ("================================================================")
	print ("optimal objective each element", optimal_values)
	print ("optimal objective sum", optimal_values_sum)

	if False:#Checing risidual for each bus
		for ele in branch:
			if ele.unknown_B:
				i = ele.from_bus
				print("unknown_bus", ele.from_bus, ele.to_bus)
				print("unknown_bus_voltage", model.v_i[ele.from_bus],value(model.v_i[ele.from_bus]), model.v_i[ele.to_bus],value(model.v_i[ele.to_bus]))
				print("unknown_bus_voltage_difference", value(model.v_i[ele.from_bus])-value(model.v_i[ele.to_bus]))
				print("unknown_branch_B", value(model.B[0]))
				current_unknown_B_from = sum(ele.calc_real_current(model.v_r[ele.from_bus], model.v_i[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.to_bus], i) for ele in branch if ele.from_bus == i)
				current_unknown_B_to = sum(ele.calc_real_current(model.v_r[ele.from_bus], model.v_i[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.to_bus], i) for ele in branch if ele.to_bus == i)
				current_unknown_B_Trans_from =sum(ele.calc_real_current(model.v_r[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.from_bus], model.v_i[ele.to_bus], i) for ele in transformer if ele.from_bus == i)
				current_unknown_B_Trans_to = sum(ele.calc_real_current(model.v_r[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.from_bus], model.v_i[ele.to_bus], i) for ele in transformer if ele.to_bus == i)
				current_unknown_B_shunt = sum(ele.calc_real_current(model.v_r[ele.Bus], model.v_i[ele.Bus]) for ele in shunt if ele.Bus == i)
				print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
				print("What is the function:","current_unknown_B_from",current_unknown_B_from)
				print("What is the function:","current_unknown_B_to",current_unknown_B_to)
				print("What is the function:","current_unknown_B_Trans_from",current_unknown_B_Trans_from)
				print("What is the function:","current_unknown_B_Trans_to",current_unknown_B_Trans_to)
				print("What is the function:","current_unknown_B_shunt",current_unknown_B_shunt)
				print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
				print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
				print("What is the value:","current_unknown_B_from",value(current_unknown_B_from))
				print("What is the value:","current_unknown_B_to",value(current_unknown_B_to))
				print("What is the value:","current_unknown_B_Trans_from",value(current_unknown_B_Trans_from))
				print("What is the value:","current_unknown_B_Trans_to",value(current_unknown_B_Trans_to))
				print("What is the value:","current_unknown_B_shunt",value(current_unknown_B_shunt))
				print("What is the residual:",(value(current_unknown_B_from) + value(current_unknown_B_to) + value(current_unknown_B_Trans_from) + value(current_unknown_B_Trans_to) + value(current_unknown_B_shunt)))
				print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
	
	
	if get_estB:
		for branches in unknown_B_between:
			(a,b,id) = branches # Branch from bus a to bus b
			
			for ele in branch:
				if ele.from_bus == a and ele.to_bus == b:
					ele.B_l = value(ele.B_l)

	if unknown_branch_B or unknown_branch_G or unknown_branch_sh:
		
		return_list = [est_vr, est_vi]

		if unknown_branch_B:
			est_B = [] # This should be a list of est of unknow B 
			
			if not delta_B:
				for ele in model.ipopt_B_list:
					est_B.append(value(model.ipopt_B_list[ele]))
					
			else:
				for ele in model.Bipopt_B_list:
					est_B.append(value(model.ipopt_B_list[ele]))
			
			print ("EST_B", est_B)
			
			return est_vr, est_vi, est_B, real_B_list, unknown_B_between, optimal_values_sum
		
		if unknown_branch_G:
			est_G = []
			for ele in model.ipopt_G_list:
					est_B.append(value(model.ipopt_G_list[ele]))
			return est_vr, est_vi, est_G, real_G, (a,b)
		
		if unknown_branch_sh:
			est_sh = []
			for ele in model.ipopt_sh_list:
				est_sh.append(value(model.ipopt_sh_list))
	
	else:
		return est_vr, est_vi,1,1,(1,1),optimal_values

	
