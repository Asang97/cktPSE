from pyomo.environ import ConcreteModel, Var, SolverFactory, Objective, ConstraintList, minimize, value
from scripts.Measure import measure_RTU
from scripts.Measure import measure_PMU

def State_estimation(v, Y_final, bus, branch, flag_WGN):
    #number of buses
	num_buses = 14
    # print ( "line Data Keys",line_data.keys())
    # Set model to be concrete
	model = ConcreteModel()

    #define measurement data
	z_RTU = measure_RTU(v, bus, branch, flag_WGN)

    # define variables
	model.v_r = Var(range(1, num_buses+1))
	model.v_i = Var(range(1, num_buses+1))
	model.n_r = Var(range(1, num_buses+1))
	model.n_i = Var(range(1, num_buses+1))
	model.n_v = Var(range(1, num_buses+1))
	model.slack_nr = Var()
    # model.n_v = Var(within=NonNegativeReals)

	counter = 0
	for ele in bus:
		counter += 1
		model.n_v[counter] = 0
		model.n_r[counter] = 0
		if counter != 1:
			model.n_i[counter] = 0
		v_r, v_i = ele.get_ipopt_vars()
		print(v_r, v_i)
		model.v_r[counter] = v_r
		model.v_i[counter] = v_i

	line_data = {}
	for ele in branch:
		line_data[ele.from_bus, ele.to_bus] = {'G': ele.G_l, 'B': ele.B_l, 'b': ele.b}
		line_data[ele.to_bus, ele.from_bus] = {'G': ele.G_l, 'B': ele.B_l, 'b': ele.b}

	print(line_data)

	unknown_G = True
	(a,b) = 1, 2
	print("Real G", line_data[a,b]["G"])
	real_G = [line_data[a,b]["G"]]
	real_B = [line_data[a,b]["B"]]
	if unknown_G: #Test the unknown G on branch compare (4,3) (3,4) and (1,2)(2,1)

		#model.G = Var()
		model.B = Var()

		# Initialize G
		#model.G = 10
		model.B = 10

		line_data[a, b]["B"] = model.B
		line_data[b, a]["B"] = model.B
		"""
      	If true, setting a G at branch (a,b) to be unknown variable G, with
      	a initial point set in model.G
     	 Note if add one unknown G is added to the problem, all two equality
      	constraints related to this with G is affected.
		Each G unknown added is replacing 2 affine equlity constraints to
 	    2 quadratic equlity constraints
		"""
	
	# There should be 4 v_r 4 v_i and 4 n_r 4 n_i in total 16 variables
	# Define the objective function to minimize noise

	# model.noise = Objective(expr=sum(model.n_r[i]**2 + model.n_i[i]**2 for i in range(2, num_buses+1))\
	#  + (model.n_r[1]**2 + model.n_i[1]**2) + (((model.v_r[1])**2-(z_RTU[1]['v_mag'])**2 ) + (model.v_i[1])**2), sense = minimize)
	w = 1000
	model.noise = Objective(expr= w*model.slack_nr**2 +  
			 			    w*sum(model.n_r[i]**2 + model.n_i[i]**2 + model.n_v[i]**2  for i in range(1, num_buses+1))\
	                        ,sense = minimize)


	model.consl = ConstraintList()
	#model.consl.add(expr=model.G==line_data[a,b]["G"])
	#model.consl.add(expr= model.G>=0)
	# Define the KCL equations
	for i in range(1, num_buses+1):
		if i != 1:
			model.consl.add(expr=(model.v_r[i]**2 + model.v_i[i]**2 - z_RTU[i]['v_mag']**2 - model.n_v[i]==0))
		else:
			model.consl.add(expr=(model.n_v[i]==0))
		"""
		This constraint is needed, with out it we can not squeeze the solution to 2
		IPOPT will find a large number of solution near the initial points
		"""
		"""
		for ele in slack:
			ele.add_slack_constraints(model)
		"""
	
		if i == 1:
			# Real voltage source
			model.consl.add(expr=model.v_r[i] - model.slack_nr - z_RTU[i]['v_mag'] ==0)
			# Imaginary voltage source is set to 0
			model.consl.add(expr=model.v_i[i] == 0)
			#
			model.consl.add(expr= sum(\
			   (model.v_r[i]-model.v_r[j])*line_data[i,j]['G']\
			     - (-model.v_i[j])*line_data[i,j]['B']\
			       for j in range(1, num_buses+1) if (i,j) in line_data.keys())\
			       #Up there should add up to all current flow at node i except I_RTU
			         + model.v_r[i]*z_RTU[i]['p']/((z_RTU[i]['v_mag'])**2)\
			         #This one should be I_RTU
			         + model.n_r[i] == 0)
			         #And the noise term

			model.consl.add(expr= sum(\
			   (model.v_r[i] - model.v_r[j])*line_data[i,j]['B']\
			     +(-model.v_i[j])*line_data[i,j]['G']\
			     + (model.v_r[i]*0.5*line_data[i,j]['b'])\
			       for j in range(1, num_buses+1) if (i,j) in line_data.keys())\
			         + model.v_r[i]*z_RTU[i]['q']/((z_RTU[i]['v_mag'])**2)\
			         + model.n_i[i] == 0)
	
		else:
			model.consl.add(expr= sum(\
			   (model.v_r[i]-model.v_r[j])*line_data[i,j]['G']\
			     - (model.v_i[i]-model.v_i[j])*line_data[i,j]['B']\
			     - (model.v_i[i]*0.5*line_data[i,j]['b'])\
			       for j in range(1, num_buses+1) if (i,j) in line_data.keys())\
			       #Up there should add up to all current flow at node i except I_RTU
			         + model.v_r[i]*z_RTU[i]['p']/((z_RTU[i]['v_mag'])**2)\
			         - model.v_i[i]*z_RTU[i]['q']/((z_RTU[i]['v_mag'])**2)\
			         #This one should be I_RTU
			         + model.n_r[i] == 0)
			         #And the noise term
	
			model.consl.add(expr= sum(\
			   (model.v_r[i] - model.v_r[j])*line_data[i,j]['B']\
			     +(model.v_i[i]-model.v_i[j])*line_data[i,j]['G']\
			     + (model.v_r[i]*0.5*line_data[i,j]['b'])\
			       for j in range(1, num_buses+1) if (i,j) in line_data.keys())\
			         + model.v_i[i]*z_RTU[i]['p']/((z_RTU[i]['v_mag'])**2)\
			         + model.v_r[i]*z_RTU[i]['q']/((z_RTU[i]['v_mag'])**2)\
			         + model.n_i[i] == 0)
		if True:
			solver = SolverFactory('ipopt')
			solver.options['tol'] = 1E-8
			solver.options["OF_hessian_approximation"]="exact"
			solver.options["max_iter"]=100
			pyomo_options = {"tee": False}
		else:
			solver = SolverFactory('multistart')
	
	# Add inequality constraints here
	result = solver.solve(model, **pyomo_options)
	# print(result)
	model.solutions.store_to(result)
	# result.write()

	est_vr= [value(model.v_r[i]) for i in range(1, num_buses+1)]
	est_vi= [value(model.v_i[i]) for i in range(1, num_buses+1)]
	if unknown_G:
		est_B = [value(model.B)]
	else:
		est_B = [0]
	print(real_B, est_B)
	"Return Estimated vr,vi,B and true value B, the position of unknown branch"
	return est_vr, est_vi, est_B, real_B, (a,b)