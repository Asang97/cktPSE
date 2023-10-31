from __future__ import division
from itertools import count
from models.Buses import Buses
import numpy as np
from typing import Union, Optional
from pyomo.environ import Var as PyomoVar

class Shunts:
	_ids = count(0)

	def __init__(self,
				 Bus,
				 G_MW,
				 B_MVAR,
				 shunt_type,
				 Vhi,
				 Vlo,
				 Bmax,
				 Bmin,
				 Binit,
				 controlBus,
				 flag_control_shunt_bus=False,
				 Nsteps=[0],
				 Bstep=[0]):

		""" Initialize a shunt in the power grid.
		Args:
			Bus (int): the bus where the shunt is located
			G_MW (float): the active component of the shunt admittance as MW per unit voltage
			B_MVAR (float): reactive component of the shunt admittance as  MVar per unit voltage
			shunt_type (int): the shunt control mode, if switched shunt
			Vhi (float): if switched shunt, the upper voltage limit
			Vlo (float): if switched shunt, the lower voltage limit
			Bmax (float): the maximum shunt susceptance possible if it is a switched shunt
			Bmin (float): the minimum shunt susceptance possible if it is a switched shunt
			Binit (float): the initial switched shunt susceptance
			controlBus (int): the bus that the shunt controls if applicable
			flag_control_shunt_bus (bool): flag that indicates if the shunt should be controlling another bus
			Nsteps (list): the number of steps by which the switched shunt should adjust itself
			Bstep (list): the admittance increase for each step in Nstep as MVar at unity voltage
		"""
		self.id = self._ids.__next__()
		self.Bus = Bus
		self.G_sh = G_MW/100
		self.B_sh = B_MVAR/100
		
		self.row_list   = np.array([])
		self.colom_list = np.array([])
		self.value_list  = np.array([])

		# You will need to implement the remainder of the __init__ function yourself.
		# You should also add some other class functions you deem necessary for stamping,
		# initializing, and processing results.

	def calc_real_current(self,
						  bus):

		Vr: Union[PyomoVar, float]
		Vi: Union[PyomoVar, float]
		
		# print(Buses.all_bus_key_[1])
		Vr = bus[Buses.all_bus_key_[self.Bus]].ipopt_vr
		Vi = bus[Buses.all_bus_key_[self.Bus]].ipopt_vi


		Ir = self.G_sh * Vr - self.B_sh * Vi
		return Ir
	
	def check_real_current(self,
						  bus):

		Vr: Union[PyomoVar, float]
		Vi: Union[PyomoVar, float]
		
		# print(Buses.all_bus_key_[1])
		Vr = (bus[Buses.all_bus_key_[self.Bus]].ipopt_vr).value
		Vi = (bus[Buses.all_bus_key_[self.Bus]].ipopt_vi).value


		Ir = self.G_sh * Vr - self.B_sh * Vi
		return Ir

	def calc_imag_current(self,
						  bus):
		
		Vr: Union[PyomoVar, float]
		Vi: Union[PyomoVar, float]
   
		# print(Buses.all_bus_key_[1])
		Vr = bus[Buses.all_bus_key_[self.Bus]].ipopt_vr
		Vi = bus[Buses.all_bus_key_[self.Bus]].ipopt_vi

		Ii = self.G_sh * Vi + self.B_sh * Vr
		return Ii
	
	def calc_real_current_measure(self,
			       				  Vr,
								  Vi):

		Ir = self.G_sh * Vr - self.B_sh * Vi
		return Ir

	def calc_imag_current_measure(self,
		                  Vr,
				          Vi):
		
		Ii = self.G_sh * Vi + self.B_sh * Vr
		return Ii

	def assign_idx(self, bus):
		self.node_Vr = bus[Buses.all_bus_key_[self.Bus]].node_Vr
		self.node_Vi = bus[Buses.all_bus_key_[self.Bus]].node_Vi

	def stampY(self, Y_matrix, row, colom, value):
		Y_matrix[row, colom] += value
	
	def stamp(self, Y_linear):
		self.stampY(Y_linear, self.node_Vr, self.node_Vr,   self.G_sh)
		self.stampY(Y_linear, self.node_Vr, self.node_Vi, -(self.B_sh))
		self.stampY(Y_linear, self.node_Vi, self.node_Vr,   self.B_sh)
		self.stampY(Y_linear, self.node_Vi, self.node_Vi,   self.G_sh)

	def sparse_stampY(self, row, colom, value):
		
		self.row_list  = np.append(self.row_list,row)
		self.colom_list = np.append(self.colom_list,colom)
		self.value_list = np.append(self.value_list,value)

	def sparse_stamp_assimbleY(self):

		self.sparse_stampY(self.node_Vr, self.node_Vr,   self.G_sh)
		self.sparse_stampY(self.node_Vr, self.node_Vi, -(self.B_sh))
		self.sparse_stampY(self.node_Vi, self.node_Vr,   self.B_sh)
		self.sparse_stampY(self.node_Vi, self.node_Vi,   self.G_sh)

		return self.row_list, self.colom_list, self.value_list


	
