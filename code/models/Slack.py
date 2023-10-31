from __future__ import division
from models.Buses import Buses
from scipy.sparse import csr_matrix
import numpy as np
from typing import Union, Optional
from pyomo.environ import Var as PyomoVar


class Slack:

	def __init__(self,
				 Bus,
				 Vset,
				 ang,
				 Pinit,
				 Qinit):
		"""Initialize slack bus in the power grid.

		Args:
			Bus (int): the bus number corresponding to the slack bus.
			Vset (float): the voltage setpoint that the slack bus must remain fixed at.
			ang (float): the slack bus voltage angle that it remains fixed at.
			Pinit (float): the initial active power that the slack bus is supplying
			Qinit (float): the initial reactive power that the slack bus is supplying
		"""
		# You will need to implement the remainder of the __init__ function yourself.

		# initialize nodes
		self.Bus = Bus
		self.node_Ir_Slack = None
		self.node_Ii_Slack = None
		self.Vset = Vset
		self.ang = ang
		
		self.Pinit_MVA = Pinit
		self.Qinit_MVA = Qinit
		self.Pinit = Pinit/100
		self.Qinit = Qinit/100
		
		self.Vr_set = Vset*np.cos(self.ang*np.pi/180)
		self.Vi_set = Vset*np.sin(self.ang*np.pi/180)
		self.Ir_init = (-self.Vr_set*self.Pinit - self.Vi_set*self.Qinit)/(Vset**2)
		self.Ii_init = (-self.Vi_set*self.Pinit + self.Vi_set*self.Qinit)/(Vset**2)
	
		self.row_list    = np.array([])
		self.colom_list  = np.array([])
		self.value_list  = np.array([])

	def __str__(self):
		return_string = 'The slack bus number is : {} with Ir_slack node as: {} and Ii_slack node as: {} '.format(self.Bus,
																												   self.node_Ir_Slack,
																												   self.node_Ii_Slack)
		return return_string


	def assign_nodes(self):
		"""Assign the additional slack bus nodes for a slack bus.

		Returns:
			None
		"""
		self.node_Ir_Slack = Buses._node_index.__next__()
		self.node_Ii_Slack = Buses._node_index.__next__()

	def assign_idx(self, bus):
		self.node_Vr = bus[Buses.all_bus_key_[self.Bus]].node_Vr
		self.node_Vi = bus[Buses.all_bus_key_[self.Bus]].node_Vi

	def stampY(self, Y_matrix, row, colom, value):
		Y_matrix[row, colom] += value

	def stampJ(self, J_matrix, colom, value):
		J_matrix[colom] += value  

	def assimble_sparse_stampY(self, Y_sparse_matrix, row_list, colom_list, value_list, size_Y):
		Y_sparse_matrix += csr_matrix ((value_list, (row_list, colom_list)), shape=(size_Y, size_Y))
		 # note the value, row, colom should be np.array form
	
	def assimble_sparse_stampJ(self, J_sparse_matrix, row_list, colom_list, value_list, size_Y):
		J_sparse_matrix += csr_matrix ((value_list, (row_list, colom_list)), shape=(size_Y, 1))

	def sparse_stampY(self, row, colom, value):
		self.row_list   = np.append(self.row_list, row)
		self.colom_list = np.append(self.colom_list, colom)
		self.value_list = np.append(self.value_list, value)
		

	# def sparse_stampJ(self, row_list, colom_list, value_list, row, value):
	#     row_list   = np.append(row_list,row)
	#     colom_list = np.append(colom_list,0.0)
	#     value_list = np.append(value_list,value)
		
	def sparse_stamp_assimbleY(self):

		self.sparse_stampY(self.node_Vr, self.node_Ir_Slack, 1.0)
		self.sparse_stampY(self.node_Vi, self.node_Ii_Slack, 1.0)
		self.sparse_stampY(self.node_Ir_Slack, self.node_Vr, 1.0)
		self.sparse_stampY(self.node_Ii_Slack, self.node_Vi, 1.0)

		return self.row_list, self.colom_list, self.value_list

	# def sparse_stamp_assimbleJ(self, J_linear ,size_Y):

	#     row_list   = np.array([])
	#     colom_list = np.array([])
	#     value_list = np.array([])

	#     self.sparse_stampJ(row_list, colom_list, value_list, self.node_Ir_Slack, self.Vset)
	#     self.sparse_stampJ(row_list, colom_list, value_list, self.node_Ii_Slack, 0)
		
	#     self.assimble_sparse_stampJ(J_linear, row_list, colom_list, value_list, size_Y)


	def stamp(self, Y_linear, J_linear):
		self.stampJ(J_linear, self.node_Ir_Slack, self.Vset)
		self.stampJ(J_linear, self.node_Ii_Slack, 0)

		self.stampY(Y_linear, self.node_Vr, self.node_Ir_Slack, 1.0)
		self.stampY(Y_linear, self.node_Vi, self.node_Ii_Slack, 1.0)
		self.stampY(Y_linear, self.node_Ir_Slack, self.node_Vr, 1.0)
		self.stampY(Y_linear, self.node_Ii_Slack, self.node_Vi, 1.0)
	
	def stamp_J(self,J_linear):
		self.stampJ(J_linear, self.node_Ir_Slack, self.Vset)
		self.stampJ(J_linear, self.node_Ii_Slack, 0)

	def calc_slack_PQ(self, V_sol):
		Ir = V_sol[self.node_Ir_Slack]
		Ii = V_sol[self.node_Ii_Slack]
		Vr = self.Vr_set
		Vi = self.Vi_set
		S = (Vr + 1j*Vi)*(Ir - 1j*Ii)
		P = -np.real(S)
		Q = np.imag(S)
		return (P, Q)

	#IPOPT related

	def create_ipopt_slack_var(self, model):

		self.ipopt_slack_nr:Union[PyomoVar, float]
		self.ipopt_slack_nr = model.ipopt_slack_nr

	def initialize_ipopt_slack_var(self):
		
		self.ipopt_slack_nr:Union[PyomoVar, float]
		self.ipopt_slack_nr.value = 0.0
	
	def add_ipopt_slack_constraint(self,
								   model,
								   vmag_mea,
								   bus):
			
		self.ipopt_slack_nr:Union[PyomoVar, float]
		ipopt_vr:Union[PyomoVar, float]
		ipopt_vi:Union[PyomoVar, float]
		
		
		ipopt_vr = bus[Buses.all_bus_key_[self.Bus]].ipopt_vr
		ipopt_vi = bus[Buses.all_bus_key_[self.Bus]].ipopt_vi

		# Real voltage source
		model.consl.add(expr= ipopt_vr - self.ipopt_slack_nr - vmag_mea ==0) #Critical change

		# Imaginary voltage source is set to 0
		model.consl.add(expr= ipopt_vi == 0)

	
		


