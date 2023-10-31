import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import scipy.sparse.linalg as solve_sparse
import timeit

class PowerFlow:

	def __init__(self,
				 case_name,
				 tol,
				 max_iters,
				 enable_limiting,
				 enable_sparse
				 ):
		"""Initialize the PowerFlow instance.

		Args:
			case_name (str): A string with the path to the test case.
			tol (float): The chosen NR tolerance.
			max_iters (int): The maximum number of NR iterations.
			enable_limiting (bool): A flag that indicates if we use voltage limiting or not in our solver.
		"""
		# Clean up the case name string
		case_name = case_name.replace('.RAW', '')
		case_name = case_name.replace('testcases/', '')

		self.case_name = case_name
		self.tol = tol
		self.max_iters = max_iters
		self.enable_limiting = enable_limiting
		self.enable_sparse = enable_sparse

	def solve(self,Y,J):
		# print(np.shape(Y))
		# print(Y)
		# print(J)
		v_new = np.linalg.solve(Y,J)
		return v_new
	
	def sparse_solve(self,Y,J):
		# print(np.shape(Y))
		# print(np.shape(J))
		# print(sparse_matrix.toarray())
		# np.savetxt("FinalStamp.csv", sparse_matrix.toarray(), delimiter=",")
		v_new=solve_sparse.spsolve(Y,J,use_umfpack=True)
		return v_new

	def apply_limiting(self, v,v_new, bus, generator, NR_k):
		
		delta=v_new-v
		dlim=0.1
		for b in bus:
			if np.absolute(delta[b.node_Vr])>=dlim:
					delta[b.node_Vr] = dlim*np.sign(delta[b.node_Vr])
			if np.absolute(delta[b.node_Vi])>=dlim:
					delta[b.node_Vi] = dlim*np.sign(delta[b.node_Vi])
		v_new = v + delta

		return v_new

	def check_error(self,v_iter_new,v_iter):
		err = np.amax(np.absolute(v_iter_new - v_iter))
		return err
	
	def stamp_sparse_linear(self, size_Y, bus, branch, slack, shunt, transformer):
		Y_sparse_linear = csr_matrix((size_Y, size_Y), dtype=np.float16)

		row_list    = np.array([])
		colom_list  = np.array([])
		value_list  = np.array([])

		for ele in branch:
			ele.assign_idx(bus)
		
		for ele in slack:
			ele.assign_idx(bus)

		for ele in shunt:
			ele.assign_idx(bus)
		
		for ele in transformer:
			ele.assign_idx(bus)

		counter = 0
		for ele in transformer:
			row_list_new, colom_list_new, value_list_new = ele.sparse_stamp_assimbleY()
			
			row_list   = np.append(row_list, row_list_new)
			colom_list = np.append(colom_list, colom_list_new)
			value_list = np.append(value_list, value_list_new)
			counter+=1

		for ele in branch:
			row_list_new, colom_list_new, value_list_new = ele.sparse_stamp_assimbleY()
			
			row_list   = np.append(row_list, row_list_new)
			colom_list = np.append(colom_list, colom_list_new)
			value_list = np.append(value_list, value_list_new)
			
		for ele in shunt:
			row_list_new, colom_list_new, value_list_new = ele.sparse_stamp_assimbleY()
			
			row_list   = np.append(row_list, row_list_new)
			colom_list = np.append(colom_list, colom_list_new)
			value_list = np.append(value_list, value_list_new)
		
	
		for ele in slack:
			row_list_new, colom_list_new, value_list_new = ele.sparse_stamp_assimbleY()
			
			row_list   = np.append(row_list, row_list_new)
			colom_list = np.append(colom_list, colom_list_new)
			value_list = np.append(value_list, value_list_new)
		
		Y_sparse_linear += csr_matrix ((value_list, (row_list, colom_list)), shape=(size_Y, size_Y))
		
		return Y_sparse_linear
	
	def stamp_linear_J(self, size_Y, bus, branch, slack, shunt, transformer):
		J_linear = np.zeros(size_Y)

		for ele in branch:
			ele.assign_idx(bus)
		
		for ele in slack:
			ele.assign_idx(bus)

		for ele in shunt:
			ele.assign_idx(bus)
		
		for ele in transformer:
			ele.assign_idx(bus)
		
		for ele in slack:
			ele.stamp_J(J_linear)
		return J_linear


	def stamp_linear(self, size_Y, bus, branch, slack, shunt, transformer):
		Y_linear = np.zeros([size_Y, size_Y])
		J_linear = np.zeros(size_Y)

		for ele in branch:
			ele.assign_idx(bus)
		
		for ele in slack:
			ele.assign_idx(bus)

		for ele in shunt:
			ele.assign_idx(bus)
		
		for ele in transformer:
			ele.assign_idx(bus)

		counter = 0
		for ele in transformer:
			ele.stamp(Y_linear, counter)
			counter+=1

		for ele in branch:
			ele.stamp(Y_linear)

		for ele in shunt:
			ele.stamp(Y_linear)
		
	
		for ele in slack:
			ele.stamp(Y_linear, J_linear)

		return Y_linear, J_linear
	
	def stamp_sparse_nonlinear(self, size_Y, bus, v_init, generator, load):
		# Y_sparse_nonlinear = csr_matrix((size_Y, size_Y), dtype=np.float16)

		row_list_nl    = np.array([])
		colom_list_nl  = np.array([])
		value_list_nl  = np.array([])

		for ele in generator:
			ele.assign_idx(bus)

		for ele in load:
			ele.assign_idx(bus)

		for ele in load:
			row_list_new, colom_list_new, value_list_new = ele.sparse_stamp_assimbleY(v_init)
			row_list_nl   = np.append(row_list_nl, row_list_new)
			colom_list_nl = np.append(colom_list_nl, colom_list_new)
			value_list_nl = np.append(value_list_nl, value_list_new)
			

		for ele in generator:
			row_list_new, colom_list_new, value_list_new = ele.sparse_stamp_assimbleY(v_init)
			
			row_list_nl   = np.append(row_list_nl, row_list_new)
			colom_list_nl = np.append(colom_list_nl, colom_list_new)
			value_list_nl = np.append(value_list_nl, value_list_new)

		Y_sparse_nonlinear = csr_matrix ((value_list_nl, (row_list_nl, colom_list_nl)), shape=(size_Y, size_Y))
		
		

		return Y_sparse_nonlinear

	def stamp_nonlinear(self, size_Y, bus, v_init, generator, load):
		Y_nonlinear = np.zeros([size_Y, size_Y])
		J_nonlinear = np.zeros(size_Y)

		for ele in generator:
			ele.assign_idx(bus)

		for ele in load:
			ele.assign_idx(bus)

		for ele in generator:
			ele.stamp(v_init, Y_nonlinear, J_nonlinear)

		for ele in load:
			ele.stamp(v_init, Y_nonlinear, J_nonlinear)


		return Y_nonlinear, J_nonlinear
		
	def stamp_nonlinear_J(self, size_Y, bus, v_init, generator, load):
		J_nonlinear = np.zeros(size_Y)
		
		for ele in generator:
			ele.assign_idx(bus)

		for ele in load:
			ele.assign_idx(bus)

		for ele in generator:
			ele.stamp_J(v_init,  J_nonlinear)

		for ele in load:
			ele.stamp_J(v_init,  J_nonlinear)

		return J_nonlinear

	def adjust_lf(self, generator, load, lf):
		
		for ele in load:
			ele.P = ele.P_origin
			ele.Q = ele.Q_origin

			ele.P = ele.P*lf
			ele.Q = ele.Q*lf

		for ele in generator:
			ele.P = ele.P_origin
			ele.P = ele.P*lf
		
		pass



	def run_powerflow(self,
					  v_init,
					  bus,
					  slack,
					  generator,
					  transformer,
					  branch,
					  shunt,
					  load,
					  size_Y,
					  enable_sparse, load_factor):
		"""Runs a positive sequence power flow using the Equivalent Circuit Formulation.

		Args:
			v_init (np.array): The initial solution vector which has the same number of rows as the Y matrix.
			bus (list): Contains all the buses in the network as instances of the Buses class.
			slack (list): Contains all the slack generators in the network as instances of the Slack class.
			generator (list): Contains all the generators in the network as instances of the Generators class.
			transformer (list): Contains all the transformers in the network as instance of the Transformers class.
			branch (list): Contains all the branches in the network as instances of the Branches class.
			shunt (list): Contains all the shunts in the network as instances of the Shunts class.
			load (list): Contains all the loads in the network as instances of the Load class.

		Returns:
			v(np.array): The final solution vector.
		"""
		# Change the PV and PQ bus power to load factor
		
		"The issue with this code is that reparse is needed to get the correct"
		"power data for the next run"
		self.adjust_lf(generator, load, load_factor)
		
		# # # Copy v_init into the Solution Vectors used during NR, v, and the final solution vector v_sol # # #
		v = np.copy(v_init)
		v_sol = np.copy(v)

		sparse_matrix = enable_sparse # Switch on or off to chose if sparse matrix is used

		# # # Stamp Linear Power Grid Elements into Y matrix # # #
		# TODO: PART 1, STEP 2.1 - Complete the stamp_linear function which stamps all linear power grid elements.
		#  This function should call the stamp_linear function of each linear element and return an updated Y matrix.
		#  You need to decide the input arguments and return values.
		
		if sparse_matrix:
			Y_sparse_linear = self.stamp_sparse_linear(size_Y, bus, branch, slack, shunt, transformer)
			# Y_linear = np.zeros([size_Y, size_Y])
			J_linear = np.zeros(size_Y)
			J_linear = self.stamp_linear_J(size_Y, bus, branch, slack, shunt, transformer)
		
		else:
			Y_linear = np.zeros([size_Y, size_Y])
			J_linear = np.zeros(size_Y)
			Y_linear, J_linear = self.stamp_linear(size_Y, bus, branch, slack, shunt, transformer)
		
		# data_df = pd.DataFrame(Y_linear)
		# data_df.to_excel("Y_lin_ini.xlsx")
		# data_df = pd.DataFrame(J_linear)
		# data_df.to_excel("J_lin_ini.xlsx")
		# self.stamp_linear(size_Y, bus, branch, slack, shunt, transformer)
		# print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
		# print (Y_linear)
		# print("J_linear",J_linear)
		

		# # # Initialize While Loop (NR) Variables # # #
		# TODO: PART 1, STEP 2.2 - Initialize the NR variables
		err_max   = 1.0  # maximum error at the current NR iteration
		tol       = self.tol  # chosen NR tolerance
		max_iters = self.max_iters
		NR_count = 0  # current NR iteration

		# # # Begin Solving Via NR # # #
		# TODO: PART 1, STEP 2.3 - Complete the NR While Loop
	   
		while err_max > tol:
			print ("ERRORMAX",err_max)
			if NR_count > max_iters:
				print("Reach maximum iteration, NO solution found")
				break
			else:
				NR_count += 1

				# # # Stamp Nonlinear Power Grid Elements into Y matrix # # #
				# TODO: PART 1, STEP 2.4 - Complete the stamp_nonlinear function which stamps all nonlinear power grid
				#  elements. This function should call the stamp_nonlinear function of each nonlinear element and return
				#  an updated Y matrix. You need to decide the input arguments and return values.
				
				
				if sparse_matrix:
					# J_sparse_nonlinear = csr_matrix((1, size_Y), dtype=np.float16)
					J_nonlinear = np.zeros(size_Y)
					Y_sparse_NL = self.stamp_sparse_nonlinear(size_Y, bus, v, generator, load) #Sparse NL, used
					Y_nonlinear, J_nonlinear = self.stamp_nonlinear(size_Y, bus, v, generator, load) #Non sparse, Y not used, J used
					Y_final = Y_sparse_linear + Y_sparse_NL
					J_final = J_linear + J_nonlinear
					
				else:
					Y_nonlinear = np.zeros([size_Y, size_Y])
					J_nonlinear = np.zeros(size_Y)
					Y_nonlinear, J_nonlinear = self.stamp_nonlinear(size_Y, bus, v, generator, load)
					Y_final = Y_linear + Y_nonlinear
					J_final = J_linear + J_nonlinear

				# if NR_count == 1:
				# 	np.savetxt('PengJ_NL_ini.csv', J_nonlinear, delimiter=',')
				# 	np.savetxt('PengY_NL_ini.csv', Y_sparse_NL.toarray(), delimiter=',')
				# 	np.savetxt('PengY_NL_ini_nospares_load.csv', Y_nonlinear, delimiter=',')

				# # # Solve The System # # #
				
				start = timeit.timeit()
				if sparse_matrix:
					v_iter_new = self.sparse_solve(Y_final,J_final)
				
				
				else:
					v_iter_new = self.solve(Y_final, J_final)
				end = timeit.timeit()
				# print("TIMETIME",end - start)

				# # # Compute The Error at the current NR iteration # # #
				# TODO: PART 1, STEP 2.6 - Finish the check_error function which calculates the maximum error, err_max
				#  You need to decide the input arguments and return values.
				err_max = self.check_error(v_iter_new, v)
				# print ("err",err_max)
			
				
				# print(" We did",NR_count, "interations")
				
				v = np.copy(v_iter_new)
				# print ("======================================================")
				# print (v)
				# print ("======================================================")

				# # # Compute The Error at the current NR iteration # # #
				# TODO: PART 2, STEP 1 - Develop the apply_limiting function which implements voltage and reactive power
				#  limiting. Also, complete the else condition. Do not complete this step until you've finished Part 1.
				#  You need to decide the input arguments and return values.
				if self.enable_limiting and err_max > tol:
					v_iter_new = self.apply_limiting(v, v_iter_new, bus, generator, NR_count)
				else:
					pass
		
		for ele in bus:
			ele.set_vsol(v_iter_new)
		
		# print ("Y_FINAL",Y_final)
		# print ("J_FINAL",J_final)
		# np.savetxt('PengJ.csv', J_final, delimiter=',')
		# np.savetxt('PengY.csv', Y_final.toarray(), delimiter=',')
		# np.savetxt('Pengv.csv', v, delimiter=',')
		return Y_final ,v
