from __future__ import division
from itertools import count
import numpy as np
from math import sqrt
from pyomo.environ import Var, ConcreteModel, ConstraintList, Objective
from typing import Union, Optional
from pyomo.environ import Var as PyomoVar

class Buses:
	_idsActiveBuses = count(1)
	_idsAllBuses = count(1)

	_node_index = count(0)
	bus_key_ = {}
	all_bus_key_ = {}

	ipopt_var_vi = {}
	ipopt_var_vr = {}

	def __init__(self,
				 Bus,
				 Type,
				 Vm_init,
				 Va_init,
				 Area):
		"""Initialize an instance of the Buses class.

		Args:
			Bus (int): The bus number.
			Type (int): The type of bus (e.g., PV, PQ, of Slack)
			Vm_init (float): The initial voltage magnitude at the bus.
			Va_init (float): The initial voltage angle at the bus.
			Area (int): The zone that the bus is in.
		"""

		self.Bus = Bus
		self.Type = Type
		self.Vm_init = Vm_init
		self.Va_init = Va_init
		self.Vr_init = Vm_init*np.cos(Va_init*np.pi/180)
		self.Vi_init = Vm_init*np.sin(Va_init*np.pi/180)
		# initialize all nodes
		self.node_Vr = None  # real voltage node at a bus
		self.node_Vi = None  # imaginary voltage node at a bus
		self.node_Q = None  # reactive power or voltage contstraint node at a bus

		#
		self.vr_sol = 1
		self.vi_sol = 0

		# To storage measurment
		self.vr_mea = 0.0
		self.vi_mea = 0.0
		self.vmag_mea = 0.0
		self.p_mea = 0.0
		self.q_mea = 0.0
		self.g_rtu = 0.0
		self.b_rtu = 0.0

		# To storage estimation

		self.vr_est = 0.0
		self.vi_est = 0.0
		self.vmag_est = 0.0
		
		# To storage current SUM on bus
		self.Ii_net = 0.0
		self.Ir_net = 0.0
		
		# flag Zero Injection
		self.flag_ZIbus = 1

		# initialize the bus key
		self.idAllBuses = self._idsAllBuses.__next__()
		Buses.all_bus_key_[self.Bus] = self.idAllBuses - 1

	def __str__(self):
		if self.Type == 1 or self.Type ==3:
			return_string = 'The bus number is : {} with Vr node as: {} and Vi node as: {} '.format(self.Bus,
																									self.node_Vr,
																									self.node_Vi)
		if self.Type == 2:
			return_string = 'The bus number is : {} with Vr node as: {} and Vi node as: {} Q node as: {}'.format(self.Bus,
																												 self.node_Vr,
																												 self.node_Vi,
																												 self.node_Q)

		return return_string
	
	def assign_nodes(self):
		"""Assign nodes based on the bus type.

		Returns:
			None
		"""
		# If Slack or PQ Bus
		if self.Type == 1 or self.Type == 3:
			self.node_Vr = self._node_index.__next__()
			self.node_Vi = self._node_index.__next__()

		# If PV Bus
		elif self.Type == 2:
			self.node_Vr = self._node_index.__next__()
			self.node_Vi = self._node_index.__next__()
			self.node_Q = self._node_index.__next__()
	
	def set_vsol(self, V):
		self.vr_sol = V[self.node_Vr]
		self.vi_sol = V[self.node_Vi]
	
	def set_ipopt_vars(self):
		self.v_r = self.vr_sol
		self.v_i = self.vi_sol

	def create_ipopt_bus_vars(self, model):

		self.ipopt_vr:Union[PyomoVar, float]
		self.ipopt_vi:Union[PyomoVar, float]
		
		#model.ipopt_vr_list[Buses.all_bus_key_[self.Bus]] = self.vr_sol
		#model.ipopt_vr_list[Buses.all_bus_key_[self.Bus]] = self.vi_sol
		self.ipopt_vr = model.ipopt_vr_list[Buses.all_bus_key_[self.Bus]]
		self.ipopt_vi = model.ipopt_vi_list[Buses.all_bus_key_[self.Bus]]

	def initialize_ipopt_bus_vars(self):
		self.ipopt_vr:Union[PyomoVar, float]
		self.ipopt_vi:Union[PyomoVar, float]
		
		#model.ipopt_vr_list[Buses.all_bus_key_[self.Bus]] = self.vr_sol
		#model.ipopt_vr_list[Buses.all_bus_key_[self.Bus]] = self.vi_sol
		self.ipopt_vr.value = self.vr_sol # model.ipopt_vr_list[Buses.all_bus_key_[self.Bus]]
		self.ipopt_vi.value = self.vi_sol #model.ipopt_vi_list[Buses.all_bus_key_[self.Bus]]
		
		return()

	
		
	def get_measurement(self, z_RTU, est_vr, est_vi):
		self.vmag_mea = z_RTU[self.Bus]['v_mag']
		self.p_mea = z_RTU[self.Bus]['p']
		self.q_mea = z_RTU[self.Bus]['q']
		print(self.Bus)
		# self.vr_est = est_vr[self.Bus-1] 
		# self.vi_est = est_vi[self.Bus-1] #dont use _+1 again, not goood blayt
		self.vr_est = est_vr[Buses.all_bus_key_[self.Bus]]
		self.vi_est = est_vi[Buses.all_bus_key_[self.Bus]]
		self.vmag_est = sqrt(self.vr_est**2 + self.vi_est**2)
		self.b_rtu = z_RTU[self.Bus]['q']/z_RTU[self.Bus]['v_mag']**2
		self.g_rtu = z_RTU[self.Bus]['q']/z_RTU[self.Bus]['v_mag']**2
	
	def add_ipopt_KCL_constraint(self,
								model,
								bus,
								branch,
								transformer,
								shunt,
								rtu,
								model_type):
		
		branch_current_real = sum(ele.calc_real_current(bus, ele.from_bus) for ele in branch if ele.from_bus == self.Bus)
		branch_current_imag = sum(ele.calc_imag_current(bus, ele.from_bus) for ele in branch if ele.from_bus == self.Bus)
		transformer_current_real = sum(ele.calc_real_current(bus, ele.from_bus) for ele in transformer if ele.from_bus == self.Bus)
		transformer_current_imag = sum(ele.calc_imag_current(bus, ele.from_bus) for ele in transformer if ele.from_bus == self.Bus)
		shunt_current_real = sum(ele.calc_real_current(bus) for ele in shunt if ele.Bus == self.Bus)
		shunt_current_imag = sum(ele.calc_imag_current(bus) for ele in shunt if ele.Bus == self.Bus)
		
		if self.flag_ZIbus == 0:
			rtu_noise_real1 = sum(self.ipopt_vr*rtu_ele.p_mea/((rtu_ele.vmag_mea)**2)\
								- self.ipopt_vi*rtu_ele.q_mea/((rtu_ele.vmag_mea)**2)
								+ rtu_ele.ipopt_nr for rtu_ele in rtu if rtu_ele.Bus == self.Bus)
			rtu_noise_imag1 = sum(self.ipopt_vi*rtu_ele.p_mea/((rtu_ele.vmag_mea)**2)\
								+ self.ipopt_vr*rtu_ele.q_mea/((rtu_ele.vmag_mea)**2)\
								+ rtu_ele.ipopt_ni for rtu_ele in rtu if rtu_ele.Bus == self.Bus)
		
			# adding for different model types
			if model_type == 4:
				model.consl.add(expr= branch_current_real\
									+ transformer_current_real\
									+ shunt_current_real\
									+ rtu_noise_real1\
									== 0)
				model.consl.add(expr= branch_current_imag\
									+ transformer_current_imag\
									+ shunt_current_imag \
									+ rtu_noise_imag1\
									== 0)
			elif model_type == 2:
				pass
			elif model_type == 3:
				pass
		
		elif self.flag_ZIbus == 1:
			print (branch_current_real\
					+ transformer_current_real\
					+ shunt_current_real)
			model.consl.add(expr= branch_current_real\
									+ transformer_current_real\
									+ shunt_current_real\
									== 0)
			model.consl.add(expr= branch_current_imag\
									+ transformer_current_imag\
									+ shunt_current_imag \
									== 0)


	def add_Mc_equality_constraint(self,
							model,
							bus,
							branch,
							transformer,
							shunt,
							rtu,
							model_type):
		
		# equality constraints

		branch_current_real = sum(ele.calc_real_current_Mc(bus, ele.from_bus) for ele in branch if ele.from_bus == self.Bus)
		branch_current_imag = sum(ele.calc_imag_current_Mc(bus, ele.from_bus) for ele in branch if ele.from_bus == self.Bus)
		transformer_current_real = sum(ele.calc_real_current(bus, ele.from_bus) for ele in transformer if ele.from_bus == self.Bus)
		transformer_current_imag = sum(ele.calc_imag_current(bus, ele.from_bus) for ele in transformer if ele.from_bus == self.Bus)
		shunt_current_real = sum(ele.calc_real_current(bus) for ele in shunt if ele.Bus == self.Bus)
		shunt_current_imag = sum(ele.calc_imag_current(bus) for ele in shunt if ele.Bus == self.Bus)
		
		rtu_noise_real1 = sum(self.ipopt_vr*rtu_ele.p_mea/((rtu_ele.vmag_mea)**2)\
							  - self.ipopt_vi*rtu_ele.q_mea/((rtu_ele.vmag_mea)**2)
							  + rtu_ele.ipopt_nr for rtu_ele in rtu if rtu_ele.Bus == self.Bus)
		rtu_noise_imag1 = sum(self.ipopt_vi*rtu_ele.p_mea/((rtu_ele.vmag_mea)**2)\
							  + self.ipopt_vr*rtu_ele.q_mea/((rtu_ele.vmag_mea)**2)\
							  + rtu_ele.ipopt_ni for rtu_ele in rtu if rtu_ele.Bus == self.Bus)
		
		# adding for different model types
		if self.flag_ZIbus == 0:

			if model_type == 4:
				model.consl.add(expr= branch_current_real\
									+ transformer_current_real\
									+ shunt_current_real\
									+ rtu_noise_real1\
									== 0)
				model.consl.add(expr= branch_current_imag\
									+ transformer_current_imag\
									+ shunt_current_imag \
									+ rtu_noise_imag1\
									== 0)
			
			elif model_type == 1:
				pass
			elif model_type == 2:
				pass
			elif model_type == 3:
				pass

		elif self.flag_ZIbus == 1:
			model.consl.add(expr= branch_current_real\
									+ transformer_current_real\
									+ shunt_current_real\
									== 0)
			model.consl.add(expr= branch_current_imag\
									+ transformer_current_imag\
									+ shunt_current_imag \
									== 0)
		
		


