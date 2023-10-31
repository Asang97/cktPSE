from parsers.parser import parse_raw
from scripts.PowerFlow import PowerFlow
from scripts.process_results import process_results
from scripts.Measure import measure_RTU, measure_PMU
from scripts.initialize import initialize
from scripts.IPOPT_workfile import State_estimation
from scripts import IPOPT_A
from scripts import mccormick
from scripts.make_df import make_table
from scripts.make_df import make_table_parameter_known
from scripts import direct_solve
from models.Buses import Buses
from models.Rtu import Rtu
from models.Transformers import Transformers
from scripts.make_figure import initialize_list, storage_data, make_plot
from scripts import KF
from scripts.kalman_filter import KalmanFilter
from scripts.options import flag_noise, RTU, PMU, origional_code, flag_McCormick, unknown_branch_G, unknown_branch_B, non_convex_equality,tighten_B,load_factor,get_estB
import matplotlib.pyplot as plt
import numpy as np
from scripts.test_B_lower import test_SBT_on_B
import time


def solve(TESTCASE, SETTINGS):
	"""Run the power flow solver.

	Args:
		TESTCASE (str): A string with the path to the test case.
		SETTINGS (dict): Contains all the solver settings in a dictionary.

	Returns:
		None
	"""
	# TODO: PART 1, STEP 0 - Initialize all the model classes in the models directory (models/) and familiarize
	#  yourself with the parameters of each model. Use the docs/DataFormats.pdf for assistance.

	# # # Parse the Test Case Data # # #
	case_name = TESTCASE
	parsed_data = parse_raw(case_name)
	
	# # # Assign Parsed Data to Variables # # #
	bus = parsed_data['buses']
	slack = parsed_data['slack']
	generator = parsed_data['generators']
	transformer = parsed_data['xfmrs']
	branch = parsed_data['branches']
	shunt = parsed_data['shunts']
	load = parsed_data['loads']

	# # # Solver Settings # # #
	tol = SETTINGS['Tolerance']  # NR solver tolerance
	max_iters = SETTINGS['Max Iters']  # maximum NR iterations
	enable_limiting = SETTINGS['Limiting']  # enable/disable voltage and reactive power limiting
	enable_sparse = SETTINGS['Sparse']


	# # # Assign System Nodes Bus by Bus # # #
	# We can use these nodes to have predetermined node number for every node in our Y matrix and J vector.
	for ele in bus:
		ele.assign_nodes()

	# Assign any slack nodes
	for ele in slack:
		ele.assign_nodes()

	for ele in transformer:
		ele.assign_nodes()

	# Check the assigned nodes for buses
	
	print("##############################################################################")
	
	for ele in slack:
		ele.assign_idx(bus)
	for ele in generator:
		ele.assign_idx(bus)
	for ele in transformer:
		ele.assign_idx(bus)
	for ele in branch:
		ele.assign_idx(bus)
	for ele in shunt:
		ele.assign_idx(bus)
	for ele in load:
		ele.assign_idx(bus)

	# # # Initialize Solution Vector - V and Q values # # #

	# determine the size of the Y matrix by looking at the total number of nodes in the system
	size_Y = Buses._node_index.__next__()
	# print ("SIZE_Y:", size_Y)
	#for GS-4 size should be 11 (bus*2 + 2*slack + 1*gen = 11)
	
	# At whitch bus the load is
	# print(load[1].Bus) 
	# The id of the certain load
	# print(load[1].id) 
   
	# TODO: PART 1, STEP 1 - Complete the function to initialize your solution vector v_init.
	v_init = np.zeros(size_Y)  # create a solution vector filled with zeros of size_Y
	initialize(v_init, bus, generator, slack) # initialize v vector with initial condition and return v_init
	# print("v_init is:",v_init)
	# # # Run Power Flow # # #
	powerflow = PowerFlow(case_name, tol, max_iters, enable_limiting, enable_sparse)

	# TODO: PART 1, STEP 2 - Complete the PowerFlow class and build your run_powerflow function to solve Equivalent
	#  Circuit Formulation powerflow. The function will return a final solution vector v. Remove run_pf and the if
	#  condition once you've finished building your solver.
	run_pf = True
	
	if run_pf:
		Y_final, v = powerflow.run_powerflow(
					  v_init,
					  bus,
					  slack,
					  generator,
					  transformer,
					  branch,
					  shunt,
					  load,
					  size_Y,
					  enable_sparse,
					  load_factor)

	# # # Process Results # # #
	process = True
	
	if process:
		process_results(v, bus, slack)

	"Measurements"
	
	flag_WGN = flag_noise

	Measure= True

	if Measure:
		if RTU:
			z0,z,intergrated_RTU_list = measure_RTU(v ,bus, branch, transformer, shunt, flag_WGN)

		rtu = intergrated_RTU_list

		
		if PMU:
			z_PMU = measure_PMU(v ,bus, branch, flag_WGN)

	"State Estimation"
	"McCor is controling if McCormick envelope is used"
	
	# test_SBT_on_B(bus, branch, transformer, shunt, slack, rtu)
	
	SE = False

	code_A = origional_code
	McCor = flag_McCormick
	# record start time
	start = time.time()
	if SE:
		if code_A:
			est_vr, est_vi, est_B, real_B, unknown_branch = IPOPT_A.State_estimation(v, Y_final, bus, branch, flag_WGN)
		
		if McCor:
			est_vr, est_vi, est_B, real_B, unknown_branch, nc_objective = mccormick.State_estimation(v, Y_final, bus, branch, flag_WGN, transformer, shunt, slack, rtu)
			if non_convex_equality:
				print("++++++++++++++++ Non-linear Equality +++++++++++++++++++")
			if tighten_B:
				print("++++++++++++++++ With SBT on B +++++++++++++++++++")
			print ("================= McCormick ===================")

		else:
			est_vr, est_vi, est_B, real_B, unknown_branch, nc_objective = State_estimation(v, Y_final, bus, branch, flag_WGN, transformer, shunt, slack, rtu)
			print ("================= non_convex ===================")
		
		#TODO Add tighten B some where after the NC
		"Consider if tighten B, we first need to find f(x*_loc) by solving the Non_convex"
		
		for ele in bus:
			ele.get_measurement(z, est_vr, est_vi)
	# record end time
	end = time.time()
	table = False

	#flag_unknown
	# all_parameter_known = False
	
	"When table=True and all_parameter_known=Flase, "
	"make sure unknown_branch_B or unknown_branch_G is ture"
	
	if table:
		print("The time of execution of above program is :",
      	(end-start) * 10**3, "ms")
		# if all_parameter_known:
		# 	(rmse_vmag_real_est,rmse_vmag_real_mea) = make_table_parameter_known(est_vr, est_vi,bus, flag_WGN)
		# 	# print(rmse_vmag_real_est,rmse_vmag_real_mea)
		if unknown_branch_G or unknown_branch_B:
			(rmse_vmag_real_est,rmse_vmag_real_mea) = make_table(est_vr, est_vi, est_B, real_B, unknown_branch, bus, flag_WGN)
		else:
			(rmse_vmag_real_est,rmse_vmag_real_mea) = make_table_parameter_known(est_vr, est_vi,bus, flag_WGN)
		
		# print ("++++++++++++++++z0+++++++++++++++++")
		# print (z0[68])
		# print ("++++++++++++++++z0+++++++++++++++++")
	
	AC_feasible_check = False
	real_current_list = []
	if AC_feasible_check:
			for bus_ele in bus:
				i = bus_ele.Bus
				real_current_RTU = + sum(bus_ele.ipopt_vr.value*rtu_ele.p_mea/((rtu_ele.vmag_mea)**2)\
								- bus_ele.ipopt_vi.value*rtu_ele.q_mea/((rtu_ele.vmag_mea)**2)\
								+ rtu_ele.ipopt_nr.value for rtu_ele in rtu if rtu_ele.Bus == bus_ele.Bus)
				# print ("RTU",real_current_RTU)
				real_currentcheck_branch = sum(ele.check_real_current(bus, ele.from_bus) for ele in branch if ele.from_bus == i) + sum(ele.check_real_current(bus, ele.to_bus) for ele in branch if ele.to_bus == i)
				# print ("branch",real_currentcheck_branch)
				real_currentcheck_trans = sum(ele.check_real_current(bus, ele.from_bus) for ele in transformer if ele.from_bus == i) + sum(ele.check_real_current(bus, ele.to_bus) for ele in transformer if ele.to_bus == i)
				real_currentcheck_shunt = sum(ele.check_real_current(bus) for ele in shunt if ele.Bus == i)
				real_currentcheck_sum = real_currentcheck_branch + real_currentcheck_trans + real_currentcheck_shunt + real_current_RTU
				real_current_list.append(real_currentcheck_sum)
			print ("real_bus_current_sum",abs(sum(real_current_list)))


	
	if get_estB:
		z0_test,z_test,intergrated_RTU_list = measure_RTU(v ,bus, branch, transformer, shunt, flag_WGN)
		# print ("++++++++++++++++z0_testB+++++++++++++++++")
		# print (z0_test[68])
		# print ("++++++++++++++++z0_testB+++++++++++++++++")

		# Check each bus
		currentcheck_list = []
		for i in range(1, len(bus)+1):
			
			flag_ZIbus = np.zeros(shape=(len(bus)+1,)) # was 119
			if abs(z0[i]['p']) <= 1e-10 and abs(z0[i]['q']) <= 1e-10:
				print ("+++++++++++++++++++++++  ZEROINJECTION  ++++++++++++++++++++++++", i)
				flag_ZIbus[i] = 1
			else:
				flag_ZIbus[i] = 0
		
			if flag_ZIbus[i]:
				currentcheck_branch = sum(ele.calc_real_current(z0_test[ele.from_bus]['vr'], z0_test[ele.from_bus]['vi'], z0_test[ele.to_bus]['vr'], z0_test[ele.to_bus]['vi'], i) for ele in branch if ele.from_bus == i)\
							+ sum(ele.calc_real_current(z0_test[ele.from_bus]['vr'], z0_test[ele.from_bus]['vi'], z0_test[ele.to_bus]['vr'], z0_test[ele.to_bus]['vi'], i) for ele in branch if ele.to_bus == i)
				
				currentcheck_trans = sum(ele.calc_real_current(z0_test[ele.from_bus]['vr'], z0_test[ele.to_bus]['vr'], z0_test[ele.from_bus]['vi'], z0_test[ele.to_bus]['vi'], i) for ele in transformer if ele.from_bus == i)\
							+ sum(ele.calc_real_current(z0_test[ele.from_bus]['vr'], z0_test[ele.to_bus]['vr'], z0_test[ele.from_bus]['vi'], z0_test[ele.to_bus]['vi'], i)for ele in transformer if ele.to_bus == i)
				
				currentcheck_shunt =  sum(ele.calc_real_current(z0_test[ele.Bus]['vr'], z0_test[ele.Bus]['vi']) for ele in shunt if ele.Bus == i)

				currentcheck_sum = currentcheck_branch + currentcheck_trans + currentcheck_shunt
				currentcheck_list.append(abs(currentcheck_sum))
				# print ("BUS",i,"currentcheck", currentcheck_sum)
			
			elif flag_ZIbus[i] == 0:
				print ("+++++++++++++++++++++++++++++++++CURRENT++++++++++++++++++++++++++++++++++++++++++++")
				# test = sum(ele.check_real_current(bus, ele.from_bus) for ele in branch if ele.from_bus == i)
				# print ("Test IS",test)
				currentcheck_branch = sum(ele.calc_real_current(bus, ele.from_bus) for ele in branch if ele.from_bus == i) + sum(ele.calc_real_current(bus, ele.to_bus) for ele in branch if ele.to_bus == i)
				# currentcheck_branch = sum(ele.calc_real_current(z0_test[ele.from_bus]['vr'], z0_test[ele.from_bus]['vi'], z0_test[ele.to_bus]['vr'], z0_test[ele.to_bus]['vi'], i) for ele in branch if ele.from_bus == i)\
				# 			+ sum(ele.calc_real_current(z0_test[ele.from_bus]['vr'], z0_test[ele.from_bus]['vi'], z0_test[ele.to_bus]['vr'], z0_test[ele.to_bus]['vi'], i) for ele in branch if ele.to_bus == i)
				
				currentcheck_trans = sum(ele.calc_real_current(z0_test[ele.from_bus]['vr'], z0_test[ele.to_bus]['vr'], z0_test[ele.from_bus]['vi'], z0_test[ele.to_bus]['vi'], i) for ele in transformer if ele.from_bus == i)\
							+ sum(ele.calc_real_current(z0_test[ele.from_bus]['vr'], z0_test[ele.to_bus]['vr'], z0_test[ele.from_bus]['vi'], z0_test[ele.to_bus]['vi'], i)for ele in transformer if ele.to_bus == i)
				
				currentcheck_shunt =  sum(ele.calc_real_current(z0_test[ele.Bus]['vr'], z0_test[ele.Bus]['vi']) for ele in shunt if ele.Bus == i)

				currentcheck_RTU = z0_test[i]['vr']*z0_test[i]['p']/((z0_test[i]['v_mag'])**2) - z0_test[i]['vi']*z0_test[i]['q']/((z0_test[i]['v_mag'])**2)

				currentcheck_sum = currentcheck_branch + currentcheck_trans + currentcheck_shunt + currentcheck_RTU
				currentcheck_list.append(abs(currentcheck_sum))
				# print ("BUS",i,"currentcheck", currentcheck_sum)
			
		print ("Max(abs(current_sum))",max(currentcheck_list))

			
			

	
	"For KF test"
	if False:
		"say we set bus1 as Vs and bus2 as Vt"
		a = 3
		b = 4
		num_mea = 20
		sol_list = []
		y_kf = KF.assimble_Y(v ,bus, branch, flag_WGN, num_mea, a, b)
		j_kf = KF.assimble_J(v ,bus, branch, flag_WGN, num_mea, a, b)
		for i in range(num_mea):
			sol = np.linalg.solve(y_kf[i],j_kf[i])
			sol_list= np.append(sol_list, 2*(np.imag(sol[1])+np.imag(sol[0])))
			# sol_list= np.append(sol_list, (np.real(sol[1])))
		
		print("SOL_LIST",(sol_list))
		
		# print("y_kf",y_kf[0])
		# print("j_kf",j_kf)
		# print("SOLUTION_Real",np.real(sol))
		# print("SOLUTION_Imag",np.imag(sol))
		
		# Define the state transition matrix, control input matrix, and observation matrix
		F = np.array([[1, 0], [0, 1]],dtype=complex)  # assume no dynamics or motion model
		B = np.array([[0], [0]],dtype=complex)  # assume no control input
		
		# observe matrix initialization
		H = np.array([[0.96848755-0.03165723j, 1.01963964+0.02711078j],
					[1.01963964+0.02711078j, 0.96848755-0.03165723j]],dtype=complex)  

		# Define the process noise covariance and measurement noise covariance matrices
		Q = np.zeros(2,dtype=complex) # assume 0 process noise
		R = np.eye(2,dtype=complex) * 0.01  # assume large measurement noise

		# Define the initial state estimate and covariance matrix
		x0 = np.array([0, 0],dtype=complex).T  # assume starting at origin
		P0 = np.eye(2,dtype=complex) * 100.0  # assume large initial uncertainty

		# Create a Kalman filter instance
		kf = KalmanFilter(F, B, H, Q, R, x0, P0)
		
		"Passing the measured data to KalmanFilter"
		kf.get_matrixs(y_kf, j_kf)

		measurements = np.array([j_kf[0],j_kf[1],j_kf[2],j_kf[3],j_kf[4],j_kf[5],j_kf[6],j_kf[7],j_kf[8],j_kf[9],\
			   j_kf[10],j_kf[11],j_kf[12],j_kf[13],j_kf[14],j_kf[15],j_kf[16],j_kf[17],j_kf[18],j_kf[19]])
		
		# Run the Kalman filter on the measurements
		"Change here to dicide which data you want output"
		filtered_states = []
		for state in kf.run_kf(measurements):
			filtered_states.append(2*np.absolute(np.imag(state[1]) + np.imag(state[0])))
			# filtered_states.append(np.absolute(np.real(state[1])))

		print ("FSTATES",filtered_states)

		# Print the filtered states
		for state in filtered_states:
			print(state)
		
		list_true = []
		for i in range (num_mea):
			# list_true.append(3.0237058538945334)
			# list_true.append(15.11852927)
			list_true.append(0.1275)

		print (np.absolute(np.imag(filtered_states)))
		print ()

		plt.plot(np.absolute((sol_list)), 'bh--', label='Direct Calculation')
		# plt.plot(np.absolute((filtered_states)), 'g^:', label='Estimates')
		plt.plot((list_true), 'r-', label='True')
		plt.legend(loc='best')
		# plt.title('Comparison of estimation and direct calculation results')
		plt.title('PMU Measurements')
		plt.xlabel('Sample Sequence Number')
		plt.ylabel('Measurement in p.u.')
		# plt.ylabel('Shunt Suseptance in p.u.')
		# plt.ylabel('Series Suseptance in p.u.')
		# plt.ylabel('Series Conductance in p.u.')
		plt.show()


	if False:
		list_rmse_est = [] 
		list_rmse_mea = []
		for i in range(29):
			z = measure_RTU(v ,bus, branch, flag_WGN)
			est_vr, est_vi, est_B, real_B, unknown_branch = State_estimation(v, Y_final, bus, branch,flag_WGN)
			for ele in bus:
				ele.get_measurement(z, est_vr, est_vi)
			(rmse_vmag_real_est,rmse_vmag_real_mea) = make_table_parameter_known(est_vr, est_vi,bus, flag_WGN)
			list_rmse_est = np.append(list_rmse_est,rmse_vmag_real_est)
			list_rmse_mea = np.append(list_rmse_mea,rmse_vmag_real_mea)
			
		plt.plot(list_rmse_est,color='green')
		plt.plot(list_rmse_mea,color='red')
		plt.show()

	"For figure making"
	make_figure = False
	if make_figure:
		(list_z_vmag, list_real_vmag, list_est_vmag) = initialize_list()
		i = 0
		"Note daily electricity usage reselution is in hours,"
		"say if we take measurement per 5min we can at most "
		"assume 12 measurement with steady state assumption"
		
		for i in range(19):
			z = measure_RTU(v ,bus, branch, flag_WGN)
			est_vr, est_vi, est_B, real_B, unknown_branch = State_estimation(v, Y_final, bus, branch,flag_WGN)
			for ele in bus:
				ele.get_measurement(z, est_vr, est_vi)
				if ele.Bus == 2:
					(list_z_vmag, list_real_vmag, list_est_vmag) = storage_data(list_z_vmag, list_real_vmag, list_est_vmag, ele)
			i+=1
		make_plot(list_z_vmag, list_real_vmag, list_est_vmag)

	"For direct calculation"
	if False:
		dc_v_matrix = direct_solve.direct_solve(direct_solve.stamp_Y(bus, branch, 8), direct_solve.stamp_J(8))
		print (dc_v_matrix)
	
	"For testing Distribudtion"
	if False:
		rmse_list = []
		for i in range (100):
			z = measure_RTU(v ,bus, branch,flag_WGN)
			est_vr, est_vi, est_B, real_B, unknown_branch = State_estimation(v, Y_final, bus, branch,flag_WGN)
			for ele in bus:
				ele.get_measurement(z, est_vr, est_vi)
			if all_parameter_known:
				(rmse_vmag_real_est,rmse_vmag_real_mea) = make_table_parameter_known(est_vr, est_vi,bus, flag_WGN)
			else:
				(rmse_vmag_real_est,rmse_vmag_real_mea) = make_table(est_vr, est_vi, est_B, real_B, unknown_branch, bus, flag_WGN)
			rmse_list = np.append(rmse_list, rmse_vmag_real_est)
		print (np.average(rmse_list))
	"For Optimal Gap"
	if False:
		est_vr_mc, est_vi_mc, est_B_mc, real_B_mc, unknown_branch_mc = mccormick.State_estimation(v, Y_final, bus, branch, flag_WGN)
		est_vr, est_vi, est_B, real_B, unknown_branch = State_estimation(v, Y_final, bus, branch, flag_WGN)
		optimal_gap = (est_B_mc[0]-est_B[0])/est_B[0]
		print (optimal_gap)

def multi_solve(TESTCASE, SETTINGS, numrepeat):
	for i in range(numrepeat):
		solve(TESTCASE, SETTINGS)
		print(i)
