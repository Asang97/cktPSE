from __future__ import division
from itertools import count
from models.Buses import Buses
import numpy as  np
import pandas as pd
from math import radians, cos, sin
from typing import Union, Optional
from pyomo.environ import Var as PyomoVar

class Transformers:
	_ids = count(0)

	def __init__(self,
				 from_bus,
				 to_bus,
				 r,
				 x,
				 status,
				 tr,
				 ang,
				 Gsh_raw,
				 Bsh_raw,
				 rating):
		"""Initialize a transformer instance

		Args:
			from_bus (int): the primary or sending end bus of the transformer.
			to_bus (int): the secondary or receiving end bus of the transformer
			r (float): the line resitance of the transformer in
			x (float): the line reactance of the transformer
			status (int): indicates if the transformer is active or not
			tr (float): transformer turns ratio
			ang (float): the phase shift angle of the transformer
			Gsh_raw (float): the shunt conductance of the transformer
			Bsh_raw (float): the shunt admittance of the transformer
			rating (float): the rating in MVA of the transformer
		"""
		self.id = self._ids.__next__()
		self.from_bus = from_bus
		self.to_bus = to_bus
		self.r = r
		self.x = x
		self.tr = tr
		self.ang = ang
		self.status = status
		self.Gsh_raw = Gsh_raw
		self.Bsh_raw = Bsh_raw
		self.G_l =  self.r/(self.r**2+self.x**2)
		self.B_l = -self.x/(self.r**2+self.x**2)
		
		# Set minimum x to avoid numerical issues.
		if abs(self.x) < 1e-6:
			if self.x < 0:
				self.x = -1e-6
			else:
				self.x = 1e-6

		self.node_from_auxr    = None
		self.node_from_auxi    = None
		self.node_to_auxr      = None
		self.node_to_auxi      = None
		self.node_Ir1_from_bus = None
		self.node_Ii1_from_bus = None
		self.node_Ir2_from_bus = None
		self.node_Ii2_from_bus = None

		self.row_list    = np.array([])
		self.colom_list  = np.array([])
		self.value_list  = np.array([])

		self.unknown_tr = False

		# You will need to implement the remainder of the __init__ function yourself.
		# You should also add some other class functions you deem necessary for stamping,
		# initializing, and processing results.
	
	def assign_nodes(self):
		"""Assign the additional aux bus nodes for a primary .

		Returns:
			None
		"""
		self.node_from_auxr    = Buses._node_index.__next__()
		self.node_from_auxi    = Buses._node_index.__next__()
		self.node_to_auxr      = Buses._node_index.__next__()
		self.node_to_auxi      = Buses._node_index.__next__()
		
		self.node_Ir1_from_bus = Buses._node_index.__next__()
		self.node_Ir2_from_bus = Buses._node_index.__next__()
		self.node_Ii1_from_bus = Buses._node_index.__next__()
		self.node_Ii2_from_bus = Buses._node_index.__next__()
		# print (self.node_from_auxr, self.node_Ii2_from_bus)
		


	def assign_idx(self, bus):
		#from_bus
		self.node_Vr_from = bus[Buses.all_bus_key_[self.from_bus]].node_Vr
		self.node_Vi_from = bus[Buses.all_bus_key_[self.from_bus]].node_Vi
	   
		#to_bus
		self.node_Vr_to   = bus[Buses.all_bus_key_[self.to_bus]].node_Vr
		self.node_Vi_to   = bus[Buses.all_bus_key_[self.to_bus]].node_Vi
	
	def stampY(self, Y_matrix, row, colom, value):
		Y_matrix[row, colom] += value

	def sparse_stampY(self, row, colom, value):
		
		self.row_list   = np.append(self.row_list, row)
		self.colom_list = np.append(self.colom_list, colom)
		self.value_list = np.append(self.value_list, value)

	def sparse_stamp_assimbleY(self):
		#stampping the real part of primary side
		self.sparse_stampY(self.node_Vr_from, self.node_Ir1_from_bus, 1)
		self.sparse_stampY( self.node_from_auxr, self.node_Ir1_from_bus, -1)
		self.sparse_stampY( self.node_Ir1_from_bus, self.node_Vr_from, 1)
		self.sparse_stampY( self.node_Ir1_from_bus, self.node_from_auxr, -1)
		self.sparse_stampY( self.node_Ir1_from_bus, self.node_to_auxr, -self.tr*np.cos(self.ang*np.pi/180))
		
		self.sparse_stampY( self.node_from_auxr, self.node_Ir2_from_bus, 1)
		self.sparse_stampY( self.node_Ir2_from_bus, self.node_from_auxr, 1)
		self.sparse_stampY( self.node_Ir2_from_bus, self.node_to_auxi, self.tr*np.sin(self.ang*np.pi/180))

		#stampping imaginary part of primary side
		self.sparse_stampY( self.node_Vi_from, self.node_Ii1_from_bus, 1)
		self.sparse_stampY( self.node_from_auxi, self.node_Ii1_from_bus, -1)
		self.sparse_stampY( self.node_Ii1_from_bus, self.node_Vi_from, 1)
		self.sparse_stampY( self.node_Ii1_from_bus, self.node_from_auxi, -1)
		self.sparse_stampY( self.node_Ii1_from_bus, self.node_to_auxr, -self.tr*np.sin(self.ang*np.pi/180))
		
		self.sparse_stampY( self.node_from_auxi, self.node_Ii2_from_bus, 1)
		self.sparse_stampY( self.node_Ii2_from_bus, self.node_from_auxi, 1)
		self.sparse_stampY( self.node_Ii2_from_bus, self.node_to_auxi, -self.tr*np.cos(self.ang*np.pi/180))

		#stamping the real part of secondary side
		self.sparse_stampY( self.node_to_auxr, self.node_Ir1_from_bus, -self.tr*np.cos(self.ang*np.pi/180))
	   
		self.sparse_stampY(self.node_to_auxr, self.node_Ii1_from_bus, -self.tr*np.sin(self.ang*np.pi/180))
		

		#stamping the imaginary part of secondary side
		self.sparse_stampY( self.node_to_auxi, self.node_Ir1_from_bus, self.tr*np.sin(self.ang*np.pi/180))
		
		self.sparse_stampY( self.node_to_auxi, self.node_Ii1_from_bus, -self.tr*np.cos(self.ang*np.pi/180))

		
		
		# Stampping loss of the transformer as transmission lines
		# Mapping the branch function to the transformer nodes
		node_Vr_from = self.node_Vr_to
		node_Vr_to   = self.node_to_auxr
		node_Vi_from = self.node_Vi_to
		node_Vi_to   = self.node_to_auxi

		self.sparse_stampY( node_Vr_from, node_Vr_from, self.G_l)
		self.sparse_stampY( node_Vi_from, node_Vi_from, self.G_l)
		self.sparse_stampY( node_Vr_to,   node_Vr_to,   self.G_l)
		self.sparse_stampY( node_Vi_to,   node_Vi_to,   self.G_l)

		self.sparse_stampY( node_Vr_from, node_Vr_to,   -(self.G_l))
		self.sparse_stampY( node_Vi_from, node_Vi_to,   -(self.G_l))
		self.sparse_stampY( node_Vr_to,   node_Vr_from, -(self.G_l))
		self.sparse_stampY( node_Vi_to,   node_Vi_from, -(self.G_l))

		self.sparse_stampY( node_Vr_from, node_Vi_to,   self.B_l)
		self.sparse_stampY( node_Vi_from, node_Vr_from, self.B_l)
		self.sparse_stampY( node_Vr_to,   node_Vi_from, self.B_l)
		self.sparse_stampY( node_Vi_to,   node_Vr_to,   self.B_l)

		self.sparse_stampY( node_Vr_from, node_Vi_from, -(self.B_l))
		self.sparse_stampY( node_Vi_from, node_Vr_to,   -(self.B_l))
		self.sparse_stampY( node_Vr_to,   node_Vi_to,   -(self.B_l))
		self.sparse_stampY( node_Vi_to,   node_Vr_from, -(self.B_l))
	   
		return self.row_list, self.colom_list, self.value_list
		

	
	def stamp(self, Y_linear, counter):

		#stampping the real part of primary side
		self.stampY(Y_linear, self.node_Vr_from, self.node_Ir1_from_bus, 1)
		self.stampY(Y_linear, self.node_from_auxr, self.node_Ir1_from_bus, -1)
		self.stampY(Y_linear, self.node_Ir1_from_bus, self.node_Vr_from, 1)
		self.stampY(Y_linear, self.node_Ir1_from_bus, self.node_from_auxr, -1)
		self.stampY(Y_linear, self.node_Ir1_from_bus, self.node_to_auxr, -self.tr*np.cos(self.ang*np.pi/180))
		
		self.stampY(Y_linear, self.node_from_auxr, self.node_Ir2_from_bus, 1)
		self.stampY(Y_linear, self.node_Ir2_from_bus, self.node_from_auxr, 1)
		self.stampY(Y_linear, self.node_Ir2_from_bus, self.node_to_auxi, self.tr*np.sin(self.ang*np.pi/180))

		#stampping imaginary part of primary side
		self.stampY(Y_linear, self.node_Vi_from, self.node_Ii1_from_bus, 1)
		self.stampY(Y_linear, self.node_from_auxi, self.node_Ii1_from_bus, -1)
		self.stampY(Y_linear, self.node_Ii1_from_bus, self.node_Vi_from, 1)
		self.stampY(Y_linear, self.node_Ii1_from_bus, self.node_from_auxi, -1)
		self.stampY(Y_linear, self.node_Ii1_from_bus, self.node_to_auxr, -self.tr*np.sin(self.ang*np.pi/180))
		
		self.stampY(Y_linear, self.node_from_auxi, self.node_Ii2_from_bus, 1)
		self.stampY(Y_linear, self.node_Ii2_from_bus, self.node_from_auxi, 1)
		self.stampY(Y_linear, self.node_Ii2_from_bus, self.node_to_auxi, -self.tr*np.cos(self.ang*np.pi/180))

		#stamping the real part of secondary side
		self.stampY(Y_linear, self.node_to_auxr, self.node_Ir1_from_bus, -self.tr*np.cos(self.ang*np.pi/180))
	   
		self.stampY(Y_linear, self.node_to_auxr, self.node_Ii1_from_bus, -self.tr*np.sin(self.ang*np.pi/180))
		

		#stamping the imaginary part of secondary side
		self.stampY(Y_linear, self.node_to_auxi, self.node_Ir1_from_bus, self.tr*np.sin(self.ang*np.pi/180))
		
		self.stampY(Y_linear, self.node_to_auxi, self.node_Ii1_from_bus, -self.tr*np.cos(self.ang*np.pi/180))

		
		
		# Stampping loss of the transformer as transmission lines
		# Mapping the branch function to the transformer nodes
		node_Vr_from = self.node_Vr_to
		node_Vr_to   = self.node_to_auxr
		node_Vi_from = self.node_Vi_to
		node_Vi_to   = self.node_to_auxi

		self.stampY(Y_linear, node_Vr_from, node_Vr_from, self.G_l)
		self.stampY(Y_linear, node_Vi_from, node_Vi_from, self.G_l)
		self.stampY(Y_linear, node_Vr_to,   node_Vr_to,   self.G_l)
		self.stampY(Y_linear, node_Vi_to,   node_Vi_to,   self.G_l)

		self.stampY(Y_linear, node_Vr_from, node_Vr_to,   -(self.G_l))
		self.stampY(Y_linear, node_Vi_from, node_Vi_to,   -(self.G_l))
		self.stampY(Y_linear, node_Vr_to,   node_Vr_from, -(self.G_l))
		self.stampY(Y_linear, node_Vi_to,   node_Vi_from, -(self.G_l))

		self.stampY(Y_linear, node_Vr_from, node_Vi_to,   self.B_l)
		self.stampY(Y_linear, node_Vi_from, node_Vr_from, self.B_l)
		self.stampY(Y_linear, node_Vr_to,   node_Vi_from, self.B_l)
		self.stampY(Y_linear, node_Vi_to,   node_Vr_to,   self.B_l)

		self.stampY(Y_linear, node_Vr_from, node_Vi_from, -(self.B_l))
		self.stampY(Y_linear, node_Vi_from, node_Vr_to,   -(self.B_l))
		self.stampY(Y_linear, node_Vr_to,   node_Vi_to,   -(self.B_l))
		self.stampY(Y_linear, node_Vi_to,   node_Vr_from, -(self.B_l))

	def create_ipopt_tr_vars(self,
							trans_to_unknowntr_key, 
			 				model):
		if self.unknown_tr == False:
				pass
		else:
			self.ipopt_tr:Union[PyomoVar, float]
			self.ipopt_tr = model.ipopt_tr_list[trans_to_unknowntr_key[self.id]]

	def initialize_ipopt_tr_vars(self):
		rand = 0.5 # Here rand should be random number
		self.ipopt_tr.value = rand*self.tr
		return()
	
	def calc_real_current(self, 
			   			  bus,
						  bus_head):
		#print ("You Came Here")
		vr_from:Union[PyomoVar, float]
		vr_to:Union[PyomoVar, float]
		vi_from:Union[PyomoVar, float]
		vi_to:Union[PyomoVar, float]
		
		# print(Buses.all_bus_key_[1])
		vr_from = bus[Buses.all_bus_key_[self.from_bus]].ipopt_vr
		vi_from = bus[Buses.all_bus_key_[self.from_bus]].ipopt_vi
		# R & I part of to bus
		vr_to = bus[Buses.all_bus_key_[self.to_bus]].ipopt_vr
		vi_to = bus[Buses.all_bus_key_[self.to_bus]].ipopt_vi



		G_cos = self.G_l * cos(radians(self.ang))
		G_sin = self.G_l * sin(radians(self.ang))
		B_cos = self.B_l * cos(radians(self.ang))
		B_sin = self.B_l * sin(radians(self.ang))
		
		if self.unknown_tr:
			inv_tr = 1/self.ipopt_tr
		else:
			inv_tr = 1/self.tr
		
		inv_tr2 = inv_tr ** 2

		Bt = self.B_l
		Mr_from = (G_cos - B_sin) * inv_tr
		Mi_from = (G_sin + B_cos) * inv_tr
		Mr_to = (G_cos + B_sin) * inv_tr
		Mi_to = (B_cos - G_sin) * inv_tr

		Ir_from = (self.G_l * inv_tr2 * vr_from) - Mr_from * vr_to - (Bt * inv_tr2) * vi_from + Mi_from * vi_to
		# Ir_from = (self.G_l * (1/self.tr）**2 * vr_from) - (self.G_l * cos(radians(self.ang)) - self.B_l * sin(radians(self.ang)))*(1/self.tr)*vr_to \
		# - (self.B_l * (1/self.tr)**2) * vi_from + (self.G_l * sin(radians(self.ang)) + self.B_l * cos(radians(self.ang))) * (1/self.tr) * vi_to
		Ir_to = -Mr_to * vr_from + self.G_l * vr_to + Mi_to * vi_from - Bt * vi_to
		# Ir_to = -(self.G_l * cos(radians(self.ang)) + self.B_l * sin(radians(self.ang))) * 1/self.tr * vr_from + self.G_l * vr_to \
		# + (self.B_l * cos(radians(self.ang)) - self.G_l * sin(radians(self.ang))) * 1/self.tr * vi_from - self.B_l * vi_to
		Ir = Ir_from if bus_head == self.from_bus else Ir_to
		return Ir

	def check_real_current(self, 
						bus,
						bus_head):

		vr_from:Union[PyomoVar, float]
		vr_to:Union[PyomoVar, float]
		vi_from:Union[PyomoVar, float]
		vi_to:Union[PyomoVar, float]
		
		# print(Buses.all_bus_key_[1])
		vr_from = (bus[Buses.all_bus_key_[self.from_bus]].ipopt_vr).value
		vi_from = (bus[Buses.all_bus_key_[self.from_bus]].ipopt_vi).value
		# R & I part of to bus
		vr_to = (bus[Buses.all_bus_key_[self.to_bus]].ipopt_vr).value
		vi_to = (bus[Buses.all_bus_key_[self.to_bus]].ipopt_vi).value



		G_cos = self.G_l * cos(radians(self.ang))
		G_sin = self.G_l * sin(radians(self.ang))
		B_cos = self.B_l * cos(radians(self.ang))
		B_sin = self.B_l * sin(radians(self.ang))
		inv_tr = 1/self.tr
		inv_tr2 = inv_tr ** 2

		Bt = self.B_l
		Mr_from = (G_cos - B_sin) * inv_tr
		Mi_from = (G_sin + B_cos) * inv_tr
		Mr_to = (G_cos + B_sin) * inv_tr
		Mi_to = (B_cos - G_sin) * inv_tr

		Ir_from = (self.G_l * inv_tr2 * vr_from) - Mr_from * vr_to - (Bt * inv_tr2) * vi_from + Mi_from * vi_to
		
		Ir_to = -Mr_to * vr_from + self.G_l * vr_to + Mi_to * vi_from - Bt * vi_to
		
		Ir = Ir_from if bus_head == self.from_bus else Ir_to
		return Ir

	def calc_imag_current(self, 
						  bus,
						  bus_head):
		
		vr_from:Union[PyomoVar, float]
		vr_to:Union[PyomoVar, float]
		vi_from:Union[PyomoVar, float]
		vi_to:Union[PyomoVar, float]
		
		# print(Buses.all_bus_key_[1])
		vr_from = bus[Buses.all_bus_key_[self.from_bus]].ipopt_vr
		vi_from = bus[Buses.all_bus_key_[self.from_bus]].ipopt_vi
		# R & I part of to bus
		vr_to = bus[Buses.all_bus_key_[self.to_bus]].ipopt_vr
		vi_to = bus[Buses.all_bus_key_[self.to_bus]].ipopt_vi

		G_cos = self.G_l * cos(radians(self.ang))
		G_sin = self.G_l * sin(radians(self.ang))
		B_cos = self.B_l * cos(radians(self.ang))
		B_sin = self.B_l * sin(radians(self.ang))
		inv_tr = 1/self.tr
		inv_tr2 = inv_tr ** 2

		Bt = self.B_l
		Mr_from = (G_cos - B_sin) * inv_tr
		Mi_from = (G_sin + B_cos) * inv_tr
		Mr_to = (G_cos + B_sin) * inv_tr
		Mi_to = (B_cos - G_sin) * inv_tr

		Ii_from = (Bt * inv_tr2) * vr_from - Mi_from * vr_to + (self.G_l * inv_tr2 * vi_from) - Mr_from * vi_to
		# Ii_from = (self.B_l * (1/self.tr）**2) * vr_from - (self.G_l * sin(radians(self.ang)) + self.B_l * cos(radians(self.ang))) * 1/self.tr * vr_to \
		# + (self.G_l * (1/self.tr)**2 * vi_from) - (self.G_l * cos(radians(self.ang)) - self.B_l * sin(radians(self.ang))) * 1/self.tr * vi_to
		Ii_to = -Mi_to * vr_from + Bt * vr_to - Mr_to * vi_from + self.G_l * vi_to
		# Ii_to = -(self.B_l * cos(radians(self.ang)) - self.G_l * sin(radians(self.ang))) * 1/self.tr * vr_from + self.B_l * vr_to\
		#  - (self.G_l * cos(radians(self.ang)) + self.B_l * sin(radians(self.ang))) * 1/self.tr * vi_from + self.G_l * vi_to
		Ii = Ii_from if bus_head == self.from_bus else Ii_to
		return Ii

	def calc_real_current_measure(self,
				   				  vr_from,
								  vr_to,
								  vi_from,
								  vi_to,
								  bus):


		G_cos = self.G_l * cos(radians(self.ang))
		G_sin = self.G_l * sin(radians(self.ang))
		B_cos = self.B_l * cos(radians(self.ang))
		B_sin = self.B_l * sin(radians(self.ang))
		inv_tr = 1/self.tr
		inv_tr2 = inv_tr ** 2

		Bt = self.B_l
		Mr_from = (G_cos - B_sin) * inv_tr
		Mi_from = (G_sin + B_cos) * inv_tr
		Mr_to = (G_cos + B_sin) * inv_tr
		Mi_to = (B_cos - G_sin) * inv_tr

		Ir_from = (self.G_l * inv_tr2 * vr_from) - Mr_from * vr_to - (Bt * inv_tr2) * vi_from + Mi_from * vi_to
		# Ir_from = (self.G_l * (1/self.tr）**2 * vr_from) - (self.G_l * cos(radians(self.ang)) - self.B_l * sin(radians(self.ang)))*(1/self.tr)*vr_to \
		# - (self.B_l * (1/self.tr)**2) * vi_from + (self.G_l * sin(radians(self.ang)) + self.B_l * cos(radians(self.ang))) * (1/self.tr) * vi_to
		Ir_to = -Mr_to * vr_from + self.G_l * vr_to + Mi_to * vi_from - Bt * vi_to
		# Ir_to = -(self.G_l * cos(radians(self.ang)) + self.B_l * sin(radians(self.ang))) * 1/self.tr * vr_from + self.G_l * vr_to \
		# + (self.B_l * cos(radians(self.ang)) - self.G_l * sin(radians(self.ang))) * 1/self.tr * vi_from - self.B_l * vi_to
		Ir = Ir_from if bus == self.from_bus else Ir_to
		return Ir

	
	def calc_imag_current_measure(self,
				 				 vr_from,
								 vr_to,
								 vi_from,
								 vi_to,  				
								 bus):
		
		G_cos = self.G_l * cos(radians(self.ang))
		G_sin = self.G_l * sin(radians(self.ang))
		B_cos = self.B_l * cos(radians(self.ang))
		B_sin = self.B_l * sin(radians(self.ang))
		inv_tr = 1/self.tr
		inv_tr2 = inv_tr ** 2

		Bt = self.B_l
		Mr_from = (G_cos - B_sin) * inv_tr
		Mi_from = (G_sin + B_cos) * inv_tr
		Mr_to = (G_cos + B_sin) * inv_tr
		Mi_to = (B_cos - G_sin) * inv_tr

		Ii_from = (Bt * inv_tr2) * vr_from - Mi_from * vr_to + (self.G_l * inv_tr2 * vi_from) - Mr_from * vi_to
		# Ii_from = (self.B_l * (1/self.tr）**2) * vr_from - (self.G_l * sin(radians(self.ang)) + self.B_l * cos(radians(self.ang))) * 1/self.tr * vr_to \
		# + (self.G_l * (1/self.tr)**2 * vi_from) - (self.G_l * cos(radians(self.ang)) - self.B_l * sin(radians(self.ang))) * 1/self.tr * vi_to
		Ii_to = -Mi_to * vr_from + Bt * vr_to - Mr_to * vi_from + self.G_l * vi_to
		# Ii_to = -(self.B_l * cos(radians(self.ang)) - self.G_l * sin(radians(self.ang))) * 1/self.tr * vr_from + self.B_l * vr_to\
		#  - (self.G_l * cos(radians(self.ang)) + self.B_l * sin(radians(self.ang))) * 1/self.tr * vi_from + self.G_l * vi_to
		Ii = Ii_from if bus == self.from_bus else Ii_to
		return Ii

	
	

		

		
