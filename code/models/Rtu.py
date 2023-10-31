from __future__ import division
from itertools import count
import numpy.random as random
from math import sqrt
from models.Buses import Buses
from pyomo.environ import Var, ConcreteModel, ConstraintList, Objective
from typing import Union, Optional
from pyomo.environ import Var as PyomoVar

class Rtu:
	_ids = count(0)
	bus_key = {}
	
	def __init__(self,
				 Bus,
				 Type,
				 Vmeas,
				 Pmeas,
				 Qmeas,
				 Noise_flag):
		"""Initialize an instance of the Buses class.

		Args:
			Bus (int): The bus number.
			Type (int): The type of bus (e.g., PV, PQ, of Slack)
			Pmeas(float): The active power measured by this RTU.
			Qmeas(float): The reactive power measured by this RTU.
			Vmeas(float): The voltage magnitude measured by this RTU.

		"""
		self.id = self._ids.__next__()
		self.Bus = Bus
		self.Type = Type

		# To storage TRUE vr and vi
		self.vr_sol = 1
		self.vi_sol = 0

		# To storage measurment
		self.vmag_mea = Vmeas
		self.p_mea = Pmeas
		self.q_mea = Qmeas
		self.g_rtu = 0
		self.b_rtu = 0

		# To storage noise flag
		self.flag_noise = Noise_flag
		Rtu.bus_key[self.Bus] = self.id


	def get_true_voltage(self,bus):
		self.vr_sol = bus[Buses.all_bus_key_[self.Bus]].vr_sol
		self.vi_sol = bus[Buses.all_bus_key_[self.Bus]].vi_sol
	
	def create_ipopt_noise_vars(self, model):
		
		self.ipopt_nr:Union[PyomoVar, float]
		self.ipopt_ni:Union[PyomoVar, float]
		self.ipopt_nv:Union[PyomoVar, float]
		
		self.ipopt_nr = model.ipopt_nr_list[self.id]
		self.ipopt_ni = model.ipopt_ni_list[self.id]
		self.ipopt_nv = model.ipopt_nv_list[self.id]

	def initialize_ipopt_bus_vars(self):
		
		self.ipopt_nr:Union[PyomoVar, float]
		self.ipopt_ni:Union[PyomoVar, float]
		self.ipopt_nv:Union[PyomoVar, float]
		
		self.ipopt_nr.value = 0.0
		self.ipopt_ni.value = 0.0
		self.ipopt_nv.value = 0.0

		