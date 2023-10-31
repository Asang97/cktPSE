from pyomo.environ import *
from scripts.Measure import measure_RTU
from scripts.Measure import measure_PMU

import numpy as np
import pandas as pd
from scripts.options import unknown_branch_G, unknown_G_between, unknown_branch_B, unknown_B_between, tolerance, max_iter, flag_tee, model_type, slack_bus_num, delta_B,get_estB\
	,w_slack, w_nv, w_nr, w_ni

"""
This test script is used to test another way of adding the constraint and
find out why ipopt is not following certain constraints
"""


def State_estimation(v, Y_final, bus, branch, flag_WGN, transformer, shunt):
	#number of buses

	i = 0
	for ele in bus:
		i = i+1
	num_buses = i

	# Set model to be concrete
	model = ConcreteModel()

	#define measurement data
	z_RTU0, z_RTU = measure_RTU(v, bus, branch, transformer, shunt, flag_WGN)

	# define variables
	model.v_r = Var(range(1, num_buses+1))
	model.v_i = Var(range(1, num_buses+1))
	model.n_r = Var(range(1, num_buses+1))
	model.n_i = Var(range(1, num_buses+1))
	#model.n_g = Var(range(1, num_buses+1))
	#model.n_b = Var(range(1, num_buses+1))
	
	model.n_v = Var(range(1, num_buses+1))
	model.slack_nr = Var()
	model.slack_nr = 0
	model.consl = ConstraintList()	

	# model.n_v = Var(within=NonNegativeReals)
	"Initializing variables"
	# for ele in bus:
		
	# 	model.n_v[ele.Bus] = 0
		
	# 	(model.v_r[ele.Bus],
	# 	model.v_i[ele.Bus], 
	# 	model.n_r[ele.Bus], 
	# 	model.n_i[ele.Bus]) =ele.initialize_ipopt_vars(
	# 							model.v_r[ele.Bus], 
	# 		    				model.v_i[ele.Bus], 
	# 							model.n_r[ele.Bus], 
	# 							model.n_i[ele.Bus])
		
	counter = 0

	# alpha = np.zeros(shape=(119,))
	# print(alpha)
	for ele in bus:
		counter += 1
		model.n_v[counter] = 0
		model.n_r[counter] = 0
		model.n_i[counter] = 0
		# if counter == 68 or counter == 116:
		# 	alpha[counter] = 1
		# else:
		# 	alpha[counter] = 1
		# if counter != 1:
		# 	model.n_i[counter] = 0
		v_r, v_i = ele.get_ipopt_vars()
		# print(v_r, v_i)
		model.v_r[counter] = v_r
		model.v_i[counter] = v_i


	line_data = {}
	for ele in branch:
		line_data[ele.from_bus, ele.to_bus] = {'G': ele.G_l, 'B': ele.B_l, 'b': ele.b}
		line_data[ele.to_bus, ele.from_bus] = {'G': ele.G_l, 'B': ele.B_l, 'b': ele.b}
	
	transformer_data = {}
	for ele in transformer:
		transformer_data[ele.from_bus, ele.to_bus] = {'G':ele.G_l, 'B':ele.B_l,'tr':ele.tr, 'ang':ele.ang}
	
	# print(line_data[3,4])

	

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
		branch_to_unknown_B = {}
		real_B_dict = {}
		model.B = Var(range(len(unknown_B_between))) # The number of unknown B is equal to the length of unknown B list

		print ("Number of Unknown B:",len(unknown_B_between))
		
		
		
		counter = 0
		real_B_list = []
		for branches in unknown_B_between:
			(a,b) = branches # Branch from bus a to bus b
			
			# Change the data in Class branch
			for ele in branch:
				if ele.from_bus == a and ele.to_bus == b:
					if delta_B:
						model.B[counter] = 0
						ele.B_l = model.B[counter] + ele.B_l_origin
					else:
						ele.B_l = model.B[counter]
						model.B[counter] = ele.B_l_origin
						# ele.add_B_ineq_constraint(model)
			
			real_B = [line_data[a,b]["B"]]
			real_B_list.append(line_data[a,b]["B"])
			
			real_B_dict.update({(a,b):real_B})
			real_B_dict.update({(b,a):real_B})
			
			branch_to_unknown_B.update({(a,b):model.B[counter]})
			branch_to_unknown_B.update({(b,a):model.B[counter]})
			
			print("Real B", line_data[a,b]["B"])
			# Initialize B
			if delta_B:
				model.B[counter] = 0
				line_data[a, b]['B'] = model.B[counter] + real_B[0]
				line_data[b, a]['B'] = model.B[counter] + real_B[0]
			else:
				model.B[counter] = real_B[0]
				"=================JUST FOR TEST DLETE AFTER USE================"
				# model.B[counter] = -10000
				"=================JUST FOR TEST DLETE AFTER USE================"
				print("Real B model", value(model.B[counter]))
				line_data[a, b]['B'] = model.B[counter]
				line_data[b, a]['B'] = model.B[counter]

			counter += 1
	else:
		pass
	
	if unknown_branch_G and unknown_branch_B:
		pass
	
	"Think WHY these Objective up there did not work well"
	"Define the objective function to minimize noise" 
 

	# model.noise = Objective(expr
	# 		 					= w_slack*(model.slack_nr**2)\
	# 		 					+ w_nv*sum(model.n_v[i]**2*alpha[i] for i in range(1, num_buses+1))\
	# 		 			    	+ w_nr*sum(model.n_r[i]**2*alpha[i] for i in range(1, num_buses+1))\
	# 	  						+ w_ni*sum(model.n_i[i]**2*alpha[i] for i in range(1, num_buses+1))\
	#                         ,sense = minimize)
	
	model.noise = Objective(expr
			 					= w_slack*(model.slack_nr**2)\
			 					+ w_nv*sum(model.n_v[i]**2 for i in range(1, num_buses+1))\
			 					+ w_nr*sum(model.n_r[i]**2 for i in range(1, num_buses+1))\
		  						+ w_ni*sum(model.n_i[i]**2 for i in range(1, num_buses+1))\
							,sense = minimize)
	

	
	
	############TESTONLY###############
	
	############TESTONLY###############
	flag_ZIbus = np.zeros(shape=(num_buses+1,))
	for i in range(1, num_buses+1):	
		if abs(z_RTU0[i]['p']) <= 1e-10 and abs(z_RTU0[i]['q']) <= 1e-10:
			print ("+++++++++++++++++++++++  ZEROINJECTION  ++++++++++++++++++++++++", i)
			flag_ZIbus[i] = 1
		else:
			flag_ZIbus[i] = 0
	
	# Define the KCL equations
	for i in range(1, num_buses+1):
	
		if i == slack_bus_num:
		
			# Real voltage source
			model.consl.add(expr=model.v_r[i] - model.slack_nr - z_RTU[i]['v_mag'] ==0) #Critical change
			# Imaginary voltage source is set to 0
			model.consl.add(expr=model.v_i[i] == 0)


		if flag_ZIbus[i]==0:
			
			if i!= slack_bus_num:
				#TODO why this constraint?
				# model.consl.add(expr=(model.v_r[i]**2 + model.v_i[i]**2 - z_RTU[i]['v_mag']**2 - model.n_v[i]==0)) # This works

				model.consl.add(expr= (model.v_r[i]**2 + model.v_i[i]**2 - (z_RTU[i]['v_mag'] - model.n_v[i])**2 == 0)) # This works 
			
				# model.consl.add(expr= ((sqrt(model.v_r[i]**2 + model.v_i[i]**2) - z_RTU[i]['v_mag']) + model.n_v[i] == 0))  # This  is the same as line above

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
						+ sum(ele.calc_imag_current(model.v_r[ele.Bus], model.v_i[ele.Bus]) for ele in shunt if ele.Bus == i)\
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
		
		elif flag_ZIbus[i]:
			
			# For Zero Injection bus, all currents add up to zero
			# Check the problem here 
			model.consl.add(expr=\
					sum(ele.calc_real_current(model.v_r[ele.from_bus], model.v_i[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.to_bus], i) for ele in branch if ele.from_bus == i)\
					+ sum(ele.calc_real_current(model.v_r[ele.from_bus], model.v_i[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.to_bus], i) for ele in branch if ele.to_bus == i)\
					+ sum(ele.calc_real_current(model.v_r[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.from_bus], model.v_i[ele.to_bus], i) for ele in transformer if ele.from_bus == i)\
					+ sum(ele.calc_real_current(model.v_r[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.from_bus], model.v_i[ele.to_bus], i) for ele in transformer if ele.to_bus == i)\
					+ sum(ele.calc_real_current(model.v_r[ele.Bus], model.v_i[ele.Bus]) for ele in shunt if ele.Bus == i)\
					# Up there should add up to all current flow at node i except I_RTU
					# No RTU Terms for ZI
					== 0)
					#And the noise term removed
			
			
			model.consl.add(expr=\
					sum(ele.calc_imag_current(model.v_r[ele.from_bus], model.v_i[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.to_bus], i) for ele in branch if ele.from_bus == i)\
					+ sum(ele.calc_imag_current(model.v_r[ele.from_bus], model.v_i[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.to_bus], i) for ele in branch if ele.to_bus == i)\
					+ sum(ele.calc_imag_current(model.v_r[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.from_bus], model.v_i[ele.to_bus], i) for ele in transformer if ele.from_bus == i)\
					+ sum(ele.calc_imag_current(model.v_r[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.from_bus], model.v_i[ele.to_bus], i) for ele in transformer if ele.to_bus == i)\
					+ sum(ele.calc_imag_current(model.v_r[ele.Bus], model.v_i[ele.Bus]) for ele in shunt if ele.Bus == i)\
					== 0)
			
			# Force related current noise terms to be 0
			model.consl.add(expr= model.n_r[i] == 0)
			model.consl.add(expr= model.n_i[i] == 0)
			model.consl.add(expr= model.n_v[i] == 0)


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
		solver.options["obj_scaling_factor"] = 1 # Same as setting w=100 for all noise 
		pyomo_options = {"tee": flag_tee}
	
	else:
		solver = SolverFactory('multistart')
	
	# Add inequality constraints here
	# model.consl.add(expr=model.G>=0)
	result = solver.solve(model, **pyomo_options)
	# print(result)
	model.solutions.store_to(result)
	# result.write()

	# Access the solver status and solution
	# print(result)
	# print(result.solver.status)
	# print(result.solver.termination_condition)

	est_vr= [value(model.v_r[i]) for i in range(1, num_buses+1)]
	est_vi= [value(model.v_i[i]) for i in range(1, num_buses+1)]

	#TODO first get the correct optiaml objective
	optimal_values_Ir = [sum(value(model.n_r[key])**2 for key in model.n_r)]
	optimal_values_Ii = [sum(value(model.n_i[key])**2 for key in model.n_i)]
	optimal_values_vmag = [sum(value(model.n_v[key])**2 for key in model.n_v)]
	optimal_values_vslack = [sum(value(model.slack_nr[key])**2 for key in model.slack_nr)]
	
	optimal_values =optimal_values_vslack + optimal_values_vmag + optimal_values_Ir + optimal_values_Ii
	optimal_values_sum = optimal_values_Ir[0] + optimal_values_Ii[0] + optimal_values_vmag[0] + optimal_values_vslack[0]
	
	print ("optimal objective each element", optimal_values)
	print ("optimal objective sum", optimal_values_sum)
	
	if get_estB:
		for branches in unknown_B_between:
			(a,b) = branches # Branch from bus a to bus b
			
			for ele in branch:
				if ele.from_bus == a and ele.to_bus == b:
					ele.B_l = value(ele.B_l)

	if unknown_branch_B:
		est_B = [] # This should be a list of unknow B 
		
		if not delta_B:
			for ele in model.B:
				est_B.append(value(model.B[ele]))
				
		else:
			for ele in model.B:
				est_B.append(value(model.B[ele]))
		
		print ("EST_B", est_B)
		
		return est_vr, est_vi, est_B, real_B_list, unknown_B_between, optimal_values_sum
	
	if unknown_branch_G:
		est_G = [value(model.G)]
		return est_vr, est_vi, est_G, real_G, (a,b)
	
	else:
		return est_vr, est_vi,1,1,(1,1),optimal_values