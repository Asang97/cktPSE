from __future__ import division
from itertools import count
from models.Buses import Buses
from scipy.sparse import csr_matrix
import numpy as np
from typing import Union, Optional
from scripts.bound_tighting import SBT_on_B_model, tighten_lower, tighten_upper, tol_check
from pyomo.environ import Var as PyomoVar, Constraint

class Branches:
	_ids = count(0)
	# For buses with the same from and to bus with different id what to do/
	def __init__(self,
				from_bus,
				to_bus,
				r,
				x,
				b,
				status,
				rateA,
				rateB,
				rateC):
		"""Initialize a branch in the power grid.

		Args:
			from_bus (int): the bus number at the sending end of the branch.
			to_bus (int): the bus number at the receiving end of the branch.
			r (float): the branch resistance
			x (float): the branch reactance
			b (float): the branch susceptance
			status (bool): indicates if the branch is online or offline
			rateA (float): The 1st rating of the line.
			rateB (float): The 2nd rating of the line
			rateC (float): The 3rd rating of the line.
		"""
		self.id = self._ids.__next__()
		self.from_bus = from_bus
		self.to_bus = to_bus
		self.r = r
		self.x = x
		self.b = b
		self.G_l =  r/(r**2 + x**2)
		self.B_l = -x/(r**2 + x**2)
		self.G_l_origin =  r/(r**2 + x**2)
		self.B_l_origin = -x/(r**2 + x**2)
		self.B_Mc_l = 0.0
		self.B_Mc_u = 0.0
		self.B_SBT_l = 0.0
		self.B_SBT_u = 0.0
		self.ipopt_wr_ini = 0.0
		self.ipopt_wi_ini = 0.0
		
		self.row_list    = np.array([])
		self.colom_list  = np.array([])
		self.value_list  = np.array([])
		
		# Class element states initial False
		self.unknown_B = False
		self.unknown_G = False
		self.unknown_sh = False
		self.tightening = False
		self.SBT_done = False

		self.tightening_B = 0.0
		self.bestknown_B = 0.0
		# To storage estimation

		self.B_est = 0.0
		self.vi_est = 0.0
		self.vmag_est = 0.0

	# You will need to implement the remainder of the __init__ function yourself.
	# You should also add some other class functions you deem necessary for stamping,
	# initializing, and processing results.
	
	# Non_convex Functions
	def create_ipopt_B_vars(self,
							branch_to_unknownB_key, 
			 				model):
		
		if self.unknown_B == False:
			pass
		else:
			self.ipopt_B:Union[PyomoVar, float]
			self.ipopt_B = model.ipopt_B_list[branch_to_unknownB_key[self.id]]

	def create_ipopt_G_vars(self,
							branch_to_unknownG_key, 
			 				model):
		
		if self.unknown_G == False:
			pass
		else:
			self.ipopt_G:Union[PyomoVar, float]
			self.ipopt_G = model.ipopt_G_list[branch_to_unknownG_key[self.id]]
	
	def create_ipopt_sh_vars(self,
							branch_to_unknownsh_key, 
			 				model):
		
		if self.unknown_sh == False:
			pass
		else:
			self.ipopt_sh:Union[PyomoVar, float]
			self.ipopt_sh = model.ipopt_sh_list[branch_to_unknownsh_key[self.id]]

	def initialize_ipopt_B_vars(self):
		
		self.ipopt_B:Union[PyomoVar, float]
		rand = 0.5 # Here rand should be random number
		self.ipopt_B.value = rand*self.B_l_origin
		
		return()
	
	def initialize_ipopt_G_vars(self):
		
		self.ipopt_G:Union[PyomoVar, float]
		rand = 0.5 # Here rand should be random number
		self.ipopt_G.value = rand*self.G_l_origin
		
		return()
	
	def initialize_ipopt_sh_vars(self):
		
		self.ipopt_sh:Union[PyomoVar, float]
		rand = 1 # Here rand should be random number
		self.ipopt_sh.value = rand*self.b
		print ("self.ipopt_sh_ini",self.ipopt_sh.value)
		return()
	
	def calc_real_current(self,
						bus,
						bus_head,
						):
		
		Vr_from: Union[PyomoVar, float]
		Vi_from: Union[PyomoVar, float]
		Vr_to:   Union[PyomoVar, float]
		Vi_to:   Union[PyomoVar, float]
		self.ipopt_B:   Union[PyomoVar, float]
		
		# self.ipopt_vr = model.ipopt_vr_list[Buses.all_bus_key_[self.Bus]]
        # self.ipopt_vi = model.ipopt_vi_list[Buses.all_bus_key_[self.Bus]]

		# print("self.from_bus",self.from_bus)
		# print("Vr_est",bus[Buses.all_bus_key_[1]].vr_est)
		
		Vr_from = bus[Buses.all_bus_key_[self.from_bus]].ipopt_vr
		Vi_from = bus[Buses.all_bus_key_[self.from_bus]].ipopt_vi
		# R & I part of to bus
		Vr_to = bus[Buses.all_bus_key_[self.to_bus]].ipopt_vr
		Vi_to = bus[Buses.all_bus_key_[self.to_bus]].ipopt_vi
		
		# Ir_from = (Vr_line * self._g_pu) - (Vi_line * self._b_pu) - (self._b_sh * 0.5 * Vi_from)
		# Ir_to = (- Vr_line * self._g_pu) + (Vi_line * self._b_pu) - (self._b_sh * 0.5 * Vi_to)
		# This function should be able to return the real part of current of certain branch
		# if self.unknown_B:
		# 	Ir_from = ((Vr_from - Vr_to)*self.G_l - (Vi_from - Vi_to)*(self.ipopt_B) - (Vi_from*0.5*self.b))
		# 	Ir_to = -(Vr_from - Vr_to)*self.G_l + (Vi_from - Vi_to)*(self.ipopt_B) - (Vi_to*0.5*self.b)
		
		# if not self.unknown_B:
		# 	Ir_from = ((Vr_from - Vr_to)*self.G_l - (Vi_from - Vi_to)*(self.B_l) - (Vi_from*0.5*self.b))
		# 	Ir_to = -(Vr_from - Vr_to)*self.G_l + (Vi_from - Vi_to)*(self.B_l) - (Vi_to*0.5*self.b)
		
		if self.unknown_B:
			B_l = self.ipopt_B
		else:
			B_l = self.B_l
		
		if self.unknown_G:
			G_l = self.ipopt_G
		else:
			G_l = self.G_l

		if self.unknown_sh:
			b_l = self.ipopt_sh
		else:
			b_l= self.b


		Ir_from = ((Vr_from - Vr_to)*G_l - (Vi_from - Vi_to)*B_l - (Vi_from*0.5*b_l))
		Ir_to = -(Vr_from - Vr_to)*G_l + (Vi_from - Vi_to)*B_l - (Vi_to*0.5*b_l)
		# print ("Bus number", self.from_bus,",", self.to_bus, "B_l",self.B_l)
		
		if bus_head == self.from_bus:
			Ir = Ir_from
		else:
			Ir = Ir_to
		# print ("Ir_calc",Ir)
		return Ir  
	
	def check_real_current(self,
						bus,
						bus_head,
						):
		
		Vr_from: Union[PyomoVar, float]
		Vi_from: Union[PyomoVar, float]
		Vr_to:   Union[PyomoVar, float]
		Vi_to:   Union[PyomoVar, float]
		self.ipopt_B:   Union[PyomoVar, float]
		
		Vr_from = (bus[Buses.all_bus_key_[self.from_bus]].ipopt_vr).value
		Vi_from = (bus[Buses.all_bus_key_[self.from_bus]].ipopt_vi).value
		# R & I part of to bus
		Vr_to = (bus[Buses.all_bus_key_[self.to_bus]].ipopt_vr).value
		Vi_to = (bus[Buses.all_bus_key_[self.to_bus]].ipopt_vi).value
		
		if self.unknown_B:
			B_l = (self.ipopt_B).value
		else:
			B_l = self.B_l
		
		if self.unknown_G:
			G_l = (self.ipopt_G).value
		else:
			G_l = self.G_l

		Ir_from = ((Vr_from - Vr_to)*G_l - (Vi_from - Vi_to)*B_l - (Vi_from*0.5*self.b))
		Ir_to = -(Vr_from - Vr_to)*G_l + (Vi_from - Vi_to)*B_l - (Vi_to*0.5*self.b)
		# print ("Bus number", self.from_bus,",", self.to_bus, "B_l",self.B_l)
		
		if bus_head == self.from_bus:
			Ir = Ir_from
		else:
			Ir = Ir_to
		# print ("Ir_check",Ir)
		return Ir  
	
	def calc_imag_current(self,
						bus,
						bus_head,
						):

		Vr_from: Union[PyomoVar, float]
		Vi_from: Union[PyomoVar, float]
		Vr_to:   Union[PyomoVar, float]
		Vi_to:   Union[PyomoVar, float]
		self.ipopt_B:   Union[PyomoVar, float]

		# print(Buses.all_bus_key_[1])
		Vr_from = bus[Buses.all_bus_key_[self.from_bus]].ipopt_vr
		Vi_from = bus[Buses.all_bus_key_[self.from_bus]].ipopt_vi
		# R & I part of to bus
		Vr_to = bus[Buses.all_bus_key_[self.to_bus]].ipopt_vr
		Vi_to = bus[Buses.all_bus_key_[self.to_bus]].ipopt_vi

		if self.unknown_B:
			B_l = self.ipopt_B
		else:
			B_l = self.B_l
		
		if self.unknown_G:
			G_l = self.ipopt_G
		else:
			G_l = self.G_l

		if self.unknown_sh:
			b_l = self.ipopt_sh
		else:
			b_l= self.b
		
        # This function should be able to return the imaginary part current of certain branch
		# Ii_from = (Vi_line * self._g_pu) + (Vr_line * self._b_pu) + (self._b_sh * 0.5 * Vr_from)
		# Ii_to = (- Vi_line * self._g_pu) - (Vr_line * self._b_pu) + (self._b_sh * 0.5 * Vr_to)
		
		if self.unknown_B: #TODO When unknown G exists, this need to be rewriten
			Ii_from =  (Vr_from - Vr_to)*B_l + (Vi_from - Vi_to)*G_l + (Vr_from*0.5*b_l)
			Ii_to =  -(Vr_from - Vr_to)*B_l - (Vi_from - Vi_to)*G_l + (Vr_to*0.5*b_l)
		else:
			Ii_from =  (Vr_from - Vr_to)*B_l + (Vi_from - Vi_to)*G_l + (Vr_from*0.5*b_l)
			Ii_to =  -(Vr_from - Vr_to)*B_l - (Vi_from - Vi_to)*G_l + (Vr_to*0.5*b_l)
		
		if bus_head == self.from_bus:
			Ii = Ii_from
		else:
			Ii = Ii_to
		
		return Ii
	

	def add_B_ineq_constraint(self,model):
		self.ipopt_B: Union[PyomoVar, float]
		if self.SBT_done:
			print("NEW INEQ is activated at BRANCH:",self.id,"With Lower:",self.B_SBT_l,"Upper:",self.B_SBT_u)
			model.consl.add(expr= self.ipopt_B <= self.B_SBT_u)
			model.consl.add(expr= self.ipopt_B >= self.B_SBT_l)
		else:
			model.consl.add(expr= self.ipopt_B <= 0.5*self.B_l_origin)
			model.consl.add(expr= self.ipopt_B >= 1.5*self.B_l_origin)

	
	# McCormick Functions 
	def create_Mc_vars(self,
							branch_to_unknownB_key,
							branch_to_unknownG_key,
							branch_to_unknownsh_key, 
			 				model):
		
		if self.unknown_B == False:
			pass
		else:
			self.ipopt_w_VrB:Union[PyomoVar, float]
			self.ipopt_w_VrB = model.ipopt_wr_list[branch_to_unknownB_key[self.id]]
			
			self.ipopt_w_ViB:Union[PyomoVar, float]
			self.ipopt_w_ViB = model.ipopt_wi_list[branch_to_unknownB_key[self.id]]
		if self.unknown_G == False:
			pass
		else:
			self.ipopt_w_VrG:Union[PyomoVar, float]
			self.ipopt_w_VrG = model.ipopt_wrG_list[branch_to_unknownG_key[self.id]]
			
			self.ipopt_w_ViG:Union[PyomoVar, float]
			self.ipopt_w_ViG = model.ipopt_wiG_list[branch_to_unknownG_key[self.id]]
				
		if self.unknown_sh == False:
			pass
		else:
			self.ipopt_w_Vi_fromsh:Union[PyomoVar, float]
			self.ipopt_w_Vi_fromsh = model.ipopt_wish_list[branch_to_unknownsh_key[self.id]]
	
	def initialize_Mc_vars(self,
							bus):
		# set upper and lower bound of B_Mc
		if self.SBT_done:
			print("NEW BOUND is activated at BRANCH:",self.id,"With Lower:",self.B_SBT_l,"Upper:",self.B_SBT_u)
			self.B_Mc_l = self.B_SBT_l
			self.B_Mc_u = self.B_SBT_u
		else:
			self.B_Mc_l = self.B_l_origin*10
			self.B_Mc_u = self.B_l_origin*0.05
		
		# print(Buses.all_bus_key_[1])
		Vr_from_sol = bus[Buses.all_bus_key_[self.from_bus]].vr_sol
		Vi_from_sol = bus[Buses.all_bus_key_[self.from_bus]].vi_sol
		# R & I part of to bus
		Vr_to_sol = bus[Buses.all_bus_key_[self.to_bus]].vr_sol
		Vi_to_sol = bus[Buses.all_bus_key_[self.to_bus]].vi_sol

		# set upper and lower bound of V_line
		# print("FROM:",self.from_bus,"TO:",self.to_bus,"Vr_from",Vr_from_sol,"Vr_to",Vr_to_sol,"Vi_from",Vi_from_sol,"Vi_to",Vi_to_sol,"Vr_diff",Vr_from_sol - Vr_to_sol,"Vi_diff",Vi_from_sol - Vi_to_sol)
		self.vi_diff_l = ((Vi_from_sol - Vi_to_sol)*0.8 if (Vi_from_sol - Vi_to_sol)>=0 else (Vi_from_sol - Vi_to_sol)*1.2)
		self.vi_diff_u = ((Vi_from_sol - Vi_to_sol)*1.2 if (Vi_from_sol - Vi_to_sol)>=0 else (Vi_from_sol - Vi_to_sol)*0.8)
		self.vr_diff_l = ((Vr_from_sol - Vr_to_sol)*0.8 if (Vr_from_sol - Vr_to_sol)>=0 else (Vr_from_sol - Vr_to_sol)*1.2)
		self.vr_diff_u = ((Vr_from_sol - Vr_to_sol)*1.2 if (Vr_from_sol - Vr_to_sol)>=0 else (Vr_from_sol - Vr_to_sol)*0.8)

		# initailize bilinear term w_VrB (Vr_line*B) and w_ViB (Vi_line*B)
		self.ipopt_w_VrB:Union[PyomoVar, float] 
		self.ipopt_w_VrB.value = 0.1 # What value shuld be given to
		
		self.ipopt_w_ViB:Union[PyomoVar, float]
		self.ipopt_w_ViB.value = 0.1 # What value shuld be given to  
		
		return()
	
		
	def calc_real_current_Mc(self,
				# Vr_from: Union[PyomoVar, float],
				# Vi_from: Union[PyomoVar, float],
				# Vr_to:   Union[PyomoVar, float],
				# Vi_to:   Union[PyomoVar, float],
				# B_l:  Union[PyomoVar, float],
				bus,
				bus_head,
				):
		
		Vr_from: Union[PyomoVar, float]
		Vi_from: Union[PyomoVar, float]
		Vr_to:   Union[PyomoVar, float]
		Vi_to:   Union[PyomoVar, float]
		# self.ipopt_B:   Union[PyomoVar, float]
	

		
		# print(Buses.all_bus_key_[1])
		Vr_from = bus[Buses.all_bus_key_[self.from_bus]].ipopt_vr
		Vi_from = bus[Buses.all_bus_key_[self.from_bus]].ipopt_vi
		# R & I part of to bus
		Vr_to = bus[Buses.all_bus_key_[self.to_bus]].ipopt_vr
		Vi_to = bus[Buses.all_bus_key_[self.to_bus]].ipopt_vi
		
		# Ir_from = (Vr_line * self._g_pu) - (Vi_line * self._b_pu) - (self._b_sh * 0.5 * Vi_from)
		# Ir_to = (- Vr_line * self._g_pu) + (Vi_line * self._b_pu) - (self._b_sh * 0.5 * Vi_to)
		
		# This function should be able to return the real part of current of certain branch
		Vr_diffGl = 0
		Vi_diffBl = 0
		Vi_frombl = 0
		Vi_tobl = 0	
		if self.unknown_B:
			Vi_diffBl = self.ipopt_w_ViB
		else:
			Vi_diffBl = (Vi_from - Vi_to)*(self.B_l)
		
		if self.unknown_G:
			Vr_diffGl = self.ipopt_w_VrG
		else:
			Vr_diffGl = (Vr_from - Vr_to)*self.G_l
		
		if self.unknown_sh:
			Vi_bl = self.ipopt_w_Vish
		else:
			if bus_head == self.from_bus:
				Vi_bl = Vi_from*self.b
			else:
				Vi_bl = Vi_to*self.b
			

		#TODO When unknown G exists, this need to be rewriten
		if self.unknown_B: # (Vi_from - Vi_to)*(self.ipopt_B) will be replaced by self.ipopt_w_ViB
			Ir_from = ((Vr_from - Vr_to)*self.G_l - self.ipopt_w_ViB - (Vi_from*0.5*self.b))
			Ir_to = -(Vr_from - Vr_to)*self.G_l + self.ipopt_w_ViB - (Vi_to*0.5*self.b)
		
		if not self.unknown_B:
			Ir_from = ((Vr_from - Vr_to)*self.G_l - (Vi_from - Vi_to)*(self.B_l) - (Vi_from*0.5*self.b))
			Ir_to = -(Vr_from - Vr_to)*self.G_l + (Vi_from - Vi_to)*(self.B_l) - (Vi_to*0.5*self.b)
		# print ("Bus number", self.from_bus,",", self.to_bus, "B_l",self.B_l)
		
		if bus_head == self.from_bus:
			Ir = Ir_from
		else:
			Ir = Ir_to
		
		return Ir
	
	def calc_imag_current_Mc(self,
				# Vr_from: Union[PyomoVar, float],
				# Vi_from: Union[PyomoVar, float],
				# Vr_to:   Union[PyomoVar, float],
				# Vi_to:   Union[PyomoVar, float],
				# B_l:  Union[PyomoVar, float],
				bus,
				bus_head,
				):

		Vr_from: Union[PyomoVar, float]
		Vi_from: Union[PyomoVar, float]
		Vr_to:   Union[PyomoVar, float]
		Vi_to:   Union[PyomoVar, float]
		self.ipopt_B:   Union[PyomoVar, float]

		# print(Buses.all_bus_key_[1])
		Vr_from = bus[Buses.all_bus_key_[self.from_bus]].ipopt_vr
		Vi_from = bus[Buses.all_bus_key_[self.from_bus]].ipopt_vi
		# R & I part of to bus
		Vr_to = bus[Buses.all_bus_key_[self.to_bus]].ipopt_vr
		Vi_to = bus[Buses.all_bus_key_[self.to_bus]].ipopt_vi
		
        # This function should be able to return the imaginary part current of certain branch
		# Ii_from = (Vi_line * self._g_pu) + (Vr_line * self._b_pu) + (self._b_sh * 0.5 * Vr_from)
		# Ii_to = (- Vi_line * self._g_pu) - (Vr_line * self._b_pu) + (self._b_sh * 0.5 * Vr_to)
		
		#TODO When unknown G exists, this need to be rewriten
		if self.unknown_B: # (Vr_from - Vr_to)*(self.ipopt_B) will be replaced by self.ipopt_w_VrB
			Ii_from =  self.ipopt_w_VrB + (Vi_from - Vi_to)*self.G_l + (Vr_from*0.5*self.b)
			Ii_to =  -self.ipopt_w_VrB - (Vi_from - Vi_to)*self.G_l + (Vr_to*0.5*self.b)
		else:
			Ii_from =  (Vr_from - Vr_to)*(self.B_l) + (Vi_from - Vi_to)*self.G_l + (Vr_from*0.5*self.b)
			Ii_to =  -(Vr_from - Vr_to)*(self.B_l) - (Vi_from - Vi_to)*self.G_l + (Vr_to*0.5*self.b)
		
		if bus_head == self.from_bus:
			Ii = Ii_from
		else:
			Ii = Ii_to
		
		return Ii

	def Mc_inequality_constraint(self,
			      				bus,
								model):
		
		Vr_from: Union[PyomoVar, float]
		Vi_from: Union[PyomoVar, float]
		Vr_to:   Union[PyomoVar, float]
		Vi_to:   Union[PyomoVar, float]
		self.ipopt_B:   Union[PyomoVar, float]
		
		# print(Buses.all_bus_key_[1])
		Vr_from = bus[Buses.all_bus_key_[self.from_bus]].ipopt_vr
		Vi_from = bus[Buses.all_bus_key_[self.from_bus]].ipopt_vi
		# R & I part of to bus
		Vr_to = bus[Buses.all_bus_key_[self.to_bus]].ipopt_vr
		Vi_to = bus[Buses.all_bus_key_[self.to_bus]].ipopt_vi
		
		print("lowerBound:",self.B_Mc_l,"Upper Bound:",self.B_Mc_u)
		# add inequality for real part
		model.consl.add(expr= (self.vi_diff_l*self.ipopt_B + (Vi_from - Vi_to)*self.B_Mc_l - self.vi_diff_l*self.B_Mc_l - self.ipopt_w_ViB <= 0))
		
		model.consl.add(expr= (self.vi_diff_u*self.ipopt_B + (Vi_from - Vi_to)*self.B_Mc_u - self.vi_diff_u*self.B_Mc_u - self.ipopt_w_ViB <= 0))
		
		model.consl.add(expr= (self.vi_diff_u*self.ipopt_B + (Vi_from - Vi_to)*self.B_Mc_l - self.vi_diff_u*self.B_Mc_l - self.ipopt_w_ViB >= 0))
		
		model.consl.add(expr= (self.vi_diff_l*self.ipopt_B + (Vi_from - Vi_to)*self.B_Mc_u - self.vi_diff_l*self.B_Mc_u - self.ipopt_w_ViB >= 0))

		model.consl.add(expr= (Vi_from - Vi_to) <= self.vi_diff_u)
		model.consl.add(expr= (Vi_from - Vi_to) >= self.vi_diff_l)

		# add inequality for imag part
		
		model.consl.add(expr= (self.vr_diff_l*self.ipopt_B + (Vr_from - Vr_to)*self.B_Mc_l - self.vr_diff_l*self.B_Mc_l - self.ipopt_w_VrB <= 0))
		
		model.consl.add(expr= (self.vr_diff_u*self.ipopt_B + (Vr_from - Vr_to)*self.B_Mc_u - self.vr_diff_u*self.B_Mc_u - self.ipopt_w_VrB <= 0))
		
		model.consl.add(expr= (self.vr_diff_u*self.ipopt_B + (Vr_from - Vr_to)*self.B_Mc_l - self.vr_diff_u*self.B_Mc_l - self.ipopt_w_VrB >= 0))
		
		model.consl.add(expr= (self.vr_diff_l*self.ipopt_B + (Vr_from - Vr_to)*self.B_Mc_u - self.vr_diff_l*self.B_Mc_u - self.ipopt_w_VrB >= 0))

		model.consl.add(expr= (Vr_from - Vr_to) <= self.vr_diff_u)
		model.consl.add(expr= (Vr_from - Vr_to) >= self.vr_diff_l)

		# add constraint for B
		model.consl.add(expr= self.ipopt_B - self.B_Mc_l >= 0)
		model.consl.add(expr= self.ipopt_B - self.B_Mc_u <= 0)
		
		vr_u = 1.1
		vr_l = 0.8
		vi_u = 0.0
		vi_l = -0.4
		"The lower and higher bound of v_r and v_i added"
		model.consl.add(expr= Vr_from - vr_u <=0)
		model.consl.add(expr= Vr_from - vr_l >=0)
		
		model.consl.add(expr= Vi_from - vi_u <=0)
		model.consl.add(expr= Vi_from - vi_l >=0)
	
	def initialize_SBT_vars(self,tighten_model):
		if self.unknown_B:
			if self.tightening == True:
				self.tightening_B: Union[PyomoVar, float]
				tighten_model.B_SBT_var: Union[PyomoVar, float]
				self.tightening_B = tighten_model.B_SBT_var
				rand = 0.8 # should be a random number
				self.bestknown_B = rand*self.B_l_origin 
				self.B_SBT_l = np.copy(self.B_Mc_l) 
				self.B_SBT_u = np.copy(self.B_Mc_u) 
				# initialize variable

	def SBT_B(self,
				tighten_model):
				#This is trying to move bound tightening in to branch so to run SBT for each branch whose branch.unknown_B is true
				# tighten_model = SBT_on_B_model(v, Y_final, bus, branch, flag_WGN, transformer, shunt, optimal_values_sum, self.B_Mc_l, self.B_Mc_u)
				
				if self.unknown_B:
					if self.tightening == True:
						self.tightening_B: Union[PyomoVar, float]
						# initialize variable
						self.tightening_B.value = self.bestknown_B
						T = 0
						while tol_check(tighten_model) == False:
						# if tol_check(tighten_model) == False:
							print ("SBT_FLAG",tol_check(tighten_model))
							print ("+++++++++++++++++++++++++++++Tighten level T:", T,"+++++++++++++++++++++++++++++++++++++++++")
							print ("AT iter:",T,"LOWER:",self.B_SBT_l,"UPPER:",self.B_SBT_u)
							T+=1
							# self.B_Mc_l = tighten_lower(tighten_model)
							# self.B_Mc_u = tighten_upper(tighten_model)
							self.B_SBT_l = tighten_lower(tighten_model)
							self.B_SBT_u = tighten_upper(tighten_model)
						if tol_check(tighten_model) == True:
							print ("If delta_B is tightened smaller than tol:",tol_check(tighten_model),"at step:",T)
					else:
						print("ERROR: ATTEMPING TO TIGHT TIGHTENED OR UNREACHED UNKNOWN BRANCH")
				else:
					print ("ERROR: ATTEMPTING TO TIGHT KNOWN BRANCH")
	
	def calc_real_current_SBT(self,
							bus,
							bus_head,
							):
		
		Vr_from: Union[PyomoVar, float]
		Vi_from: Union[PyomoVar, float]
		Vr_to:   Union[PyomoVar, float]
		Vi_to:   Union[PyomoVar, float]
		self.tightening_B:   Union[PyomoVar, float]
		
		# self.ipopt_vr = model.ipopt_vr_list[Buses.all_bus_key_[self.Bus]]
        # self.ipopt_vi = model.ipopt_vi_list[Buses.all_bus_key_[self.Bus]]

		
		# print(Buses.all_bus_key_[1])
		Vr_from = bus[Buses.all_bus_key_[self.from_bus]].ipopt_vr
		Vi_from = bus[Buses.all_bus_key_[self.from_bus]].ipopt_vi
		# R & I part of to bus
		Vr_to = bus[Buses.all_bus_key_[self.to_bus]].ipopt_vr
		Vi_to = bus[Buses.all_bus_key_[self.to_bus]].ipopt_vi
		
		# Ir_from = (Vr_line * self._g_pu) - (Vi_line * self._b_pu) - (self._b_sh * 0.5 * Vi_from)
		# Ir_to = (- Vr_line * self._g_pu) + (Vi_line * self._b_pu) - (self._b_sh * 0.5 * Vi_to)
		# This function should be able to return the real part of current of certain branch
		if self.unknown_B and self.tightening:
			print (type(self.tightening_B))
			Ir_from = ((Vr_from - Vr_to)*self.G_l - self.ipopt_w_ViB - (Vi_from*0.5*self.b))
			Ir_to = -(Vr_from - Vr_to)*self.G_l + self.ipopt_w_ViB - (Vi_to*0.5*self.b)
		elif self.unknown_B and not self.tightening:
			Ir_from = ((Vr_from - Vr_to)*self.G_l - (Vi_from - Vi_to)*(self.bestknown_B) - (Vi_from*0.5*self.b))
			Ir_to = -(Vr_from - Vr_to)*self.G_l + (Vi_from - Vi_to)*(self.bestknown_B) - (Vi_to*0.5*self.b)
		elif not self.unknown_B and not self.tightening:
			Ir_from = ((Vr_from - Vr_to)*self.G_l - (Vi_from - Vi_to)*(self.B_l) - (Vi_from*0.5*self.b))
			Ir_to = -(Vr_from - Vr_to)*self.G_l + (Vi_from - Vi_to)*(self.B_l) - (Vi_to*0.5*self.b)
		else:
			print("ERROR: TIGHTING OVER KNOWN BRANCH")
		
		if bus_head == self.from_bus:
			Ir = Ir_from
		else:
			Ir = Ir_to

		return Ir  
	
	def calc_imag_current_SBT(self,
						bus,
						bus_head,
						):

		Vr_from: Union[PyomoVar, float]
		Vi_from: Union[PyomoVar, float]
		Vr_to:   Union[PyomoVar, float]
		Vi_to:   Union[PyomoVar, float]
		self.tightening_B:   Union[PyomoVar, float]

		# print(Buses.all_bus_key_[1])
		Vr_from = bus[Buses.all_bus_key_[self.from_bus]].ipopt_vr
		Vi_from = bus[Buses.all_bus_key_[self.from_bus]].ipopt_vi
		# R & I part of to bus
		Vr_to = bus[Buses.all_bus_key_[self.to_bus]].ipopt_vr
		Vi_to = bus[Buses.all_bus_key_[self.to_bus]].ipopt_vi
		
        # This function should be able to return the imaginary part current of certain branch
		# Ii_from = (Vi_line * self._g_pu) + (Vr_line * self._b_pu) + (self._b_sh * 0.5 * Vr_from)
		# Ii_to = (- Vi_line * self._g_pu) - (Vr_line * self._b_pu) + (self._b_sh * 0.5 * Vr_to)
		
		if self.unknown_B and self.tightening:
			Ii_from =  self.ipopt_w_VrB + (Vi_from - Vi_to)*self.G_l + (Vr_from*0.5*self.b)
			Ii_to =  -self.ipopt_w_VrB - (Vi_from - Vi_to)*self.G_l + (Vr_to*0.5*self.b)
		elif self.unknown_B and not self.tightening:
			Ii_from =  (Vr_from - Vr_to)*(self.bestknown_B) + (Vi_from - Vi_to)*self.G_l + (Vr_from*0.5*self.b)
			Ii_to =  -(Vr_from - Vr_to)*(self.bestknown_B) - (Vi_from - Vi_to)*self.G_l + (Vr_to*0.5*self.b)
		elif not self.unknown_B and not self.tightening:
			Ii_from =  (Vr_from - Vr_to)*(self.B_l) + (Vi_from - Vi_to)*self.G_l + (Vr_from*0.5*self.b)
			Ii_to =  -(Vr_from - Vr_to)*(self.B_l) - (Vi_from - Vi_to)*self.G_l + (Vr_to*0.5*self.b)
		else:
			print("ERROR: TIGHTING OVER KNOWN BRANCH")
		
		if bus_head == self.from_bus:
			Ii = Ii_from
		else:
			Ii = Ii_to
		
		return Ii
	
	def Mc_SBT_inequality_constraint(self,
			      				bus,
								model):
		
		Vr_from: Union[PyomoVar, float]
		Vi_from: Union[PyomoVar, float]
		Vr_to:   Union[PyomoVar, float]
		Vi_to:   Union[PyomoVar, float]
		self.tightening_B: Union[PyomoVar, float]
		
		# print(Buses.all_bus_key_[1])
		Vr_from = bus[Buses.all_bus_key_[self.from_bus]].ipopt_vr
		Vi_from = bus[Buses.all_bus_key_[self.from_bus]].ipopt_vi
		# R & I part of to bus
		Vr_to = bus[Buses.all_bus_key_[self.to_bus]].ipopt_vr
		Vi_to = bus[Buses.all_bus_key_[self.to_bus]].ipopt_vi
		
		if self.tightening:
			print("lowerBound:",self.B_Mc_l,"Upper Bound:",self.B_Mc_u)
			# add inequality for real part
			model.consl.add(expr= (self.vi_diff_l*self.tightening_B + (Vi_from - Vi_to)*self.B_Mc_l - self.vi_diff_l*self.B_Mc_l - self.ipopt_w_ViB <= 0))
			
			model.consl.add(expr= (self.vi_diff_u*self.tightening_B + (Vi_from - Vi_to)*self.B_Mc_u - self.vi_diff_u*self.B_Mc_u - self.ipopt_w_ViB <= 0))
			
			model.consl.add(expr= (self.vi_diff_u*self.tightening_B + (Vi_from - Vi_to)*self.B_Mc_l - self.vi_diff_u*self.B_Mc_l - self.ipopt_w_ViB >= 0))
			
			model.consl.add(expr= (self.vi_diff_l*self.tightening_B + (Vi_from - Vi_to)*self.B_Mc_u - self.vi_diff_l*self.B_Mc_u - self.ipopt_w_ViB >= 0))

			model.consl.add(expr= (Vi_from - Vi_to) <= self.vi_diff_u)
			model.consl.add(expr= (Vi_from - Vi_to) >= self.vi_diff_l)

			# add inequality for imag part
			
			model.consl.add(expr= (self.vr_diff_l*self.tightening_B + (Vr_from - Vr_to)*self.B_Mc_l - self.vr_diff_l*self.B_Mc_l - self.ipopt_w_VrB <= 0))
			
			model.consl.add(expr= (self.vr_diff_u*self.tightening_B + (Vr_from - Vr_to)*self.B_Mc_u - self.vr_diff_u*self.B_Mc_u - self.ipopt_w_VrB <= 0))
			
			model.consl.add(expr= (self.vr_diff_u*self.tightening_B + (Vr_from - Vr_to)*self.B_Mc_l - self.vr_diff_u*self.B_Mc_l - self.ipopt_w_VrB >= 0))
			
			model.consl.add(expr= (self.vr_diff_l*self.tightening_B + (Vr_from - Vr_to)*self.B_Mc_u - self.vr_diff_l*self.B_Mc_u - self.ipopt_w_VrB >= 0))

			model.consl.add(expr= (Vr_from - Vr_to) <= self.vr_diff_u)
			model.consl.add(expr= (Vr_from - Vr_to) >= self.vr_diff_l)

			# add constraint for B
			print ("At branch:",self.id,type(self.tightening_B))
			model.consl.add(expr= self.tightening_B - self.B_Mc_l >= 0)
			model.consl.add(expr= self.tightening_B - self.B_Mc_u <= 0)
			
			vr_u = 1.1
			vr_l = 0.8
			vi_u = 0.0
			vi_l = -0.4
			"The lower and higher bound of v_r and v_i added"
			model.consl.add(expr= Vr_from - vr_u <=0)
			model.consl.add(expr= Vr_from - vr_l >=0)
			
			model.consl.add(expr= Vi_from - vi_u <=0)
			model.consl.add(expr= Vi_from - vi_l >=0)
	
	# Powerflow functions 
	def stampY(self, Y_matrix, row, colom, value):
		Y_matrix[row, colom] += value

	def assimble_sparse_stampY(self, Y_sparse_matrix, row_list, colom_list, value_list, size_Y):
		Y_sparse_matrix += csr_matrix ((value_list, (row_list, colom_list)), shape=(size_Y, size_Y)) # note the value, row, colom should be np.array form
	
	def sparse_stampY(self, row, colom, value):
		
		self.row_list  = np.append(self.row_list, row)
		self.colom_list = np.append(self.colom_list, colom)
		self.value_list = np.append(self.value_list, value)
		

	def assign_idx(self, bus):
		# R & I part of from bus
		# print(self.from_bus)
		# print(Buses.all_bus_key_[1])
		self.node_Vr_from = bus[Buses.all_bus_key_[self.from_bus]].node_Vr
		self.node_Vi_from = bus[Buses.all_bus_key_[self.from_bus]].node_Vi
		# R & I part of to bus
		self.node_Vr_to = bus[Buses.all_bus_key_[self.to_bus]].node_Vr
		self.node_Vi_to = bus[Buses.all_bus_key_[self.to_bus]].node_Vi

	def sparse_stamp_assimbleY(self): #Forming the list of row, colom, value to creat the CSR_matrix
		
		
		self.sparse_stampY(self.node_Vr_from, self.node_Vr_from, self.G_l)
		self.sparse_stampY(self.node_Vi_from, self.node_Vi_from, self.G_l )
		self.sparse_stampY(self.node_Vr_to, self.node_Vr_to, self.G_l )
		self.sparse_stampY(self.node_Vi_to, self.node_Vi_to, self.G_l )

		self.sparse_stampY(self.node_Vr_from, self.node_Vr_to, -(self.G_l) )
		self.sparse_stampY(self.node_Vi_from, self.node_Vi_to, -(self.G_l) )
		self.sparse_stampY(self.node_Vr_to, self.node_Vr_from, -(self.G_l) )
		self.sparse_stampY(self.node_Vi_to, self.node_Vi_from, -(self.G_l) )

		self.sparse_stampY(self.node_Vr_from, self.node_Vi_to, self.B_l )
		self.sparse_stampY(self.node_Vi_from, self.node_Vr_from, self.B_l )
		self.sparse_stampY(self.node_Vr_to, self.node_Vi_from, self.B_l )
		self.sparse_stampY(self.node_Vi_to, self.node_Vr_to, self.B_l )

		self.sparse_stampY(self.node_Vr_from, self.node_Vi_from, -(self.B_l) )
		self.sparse_stampY(self.node_Vi_from, self.node_Vr_to, -(self.B_l) )
		self.sparse_stampY(self.node_Vr_to, self.node_Vi_to, -(self.B_l) )
		self.sparse_stampY(self.node_Vi_to, self.node_Vr_from, -(self.B_l) )

		self.sparse_stampY(self.node_Vr_from, self.node_Vi_from, -(self.b)/2 )
		self.sparse_stampY(self.node_Vr_to,   self.node_Vi_to, -(self.b)/2 )
		self.sparse_stampY(self.node_Vi_from, self.node_Vr_from, (self.b)/2 )
		self.sparse_stampY(self.node_Vi_to,   self.node_Vr_to, (self.b)/2 )
		
		return self.row_list, self.colom_list, self.value_list
		

	def stamp(self, Y_linear):
		
		self.stampY(Y_linear, self.node_Vr_from, self.node_Vr_from, self.G_l)
		self.stampY(Y_linear, self.node_Vi_from, self.node_Vi_from, self.G_l)
		self.stampY(Y_linear, self.node_Vr_to, self.node_Vr_to, self.G_l)
		self.stampY(Y_linear, self.node_Vi_to, self.node_Vi_to, self.G_l)

		self.stampY(Y_linear, self.node_Vr_from, self.node_Vr_to, -(self.G_l))
		self.stampY(Y_linear, self.node_Vi_from, self.node_Vi_to, -(self.G_l))
		self.stampY(Y_linear, self.node_Vr_to, self.node_Vr_from, -(self.G_l))
		self.stampY(Y_linear, self.node_Vi_to, self.node_Vi_from, -(self.G_l))

		self.stampY(Y_linear, self.node_Vr_from, self.node_Vi_to, self.B_l)
		self.stampY(Y_linear, self.node_Vi_from, self.node_Vr_from, self.B_l)
		self.stampY(Y_linear, self.node_Vr_to, self.node_Vi_from, self.B_l)
		self.stampY(Y_linear, self.node_Vi_to, self.node_Vr_to, self.B_l)

		self.stampY(Y_linear, self.node_Vr_from, self.node_Vi_from, -(self.B_l))
		self.stampY(Y_linear, self.node_Vi_from, self.node_Vr_to, -(self.B_l))
		self.stampY(Y_linear, self.node_Vr_to, self.node_Vi_to, -(self.B_l))
		self.stampY(Y_linear, self.node_Vi_to, self.node_Vr_from, -(self.B_l))

		self.stampY(Y_linear, self.node_Vr_from, self.node_Vi_from, -(self.b)/2)
		self.stampY(Y_linear, self.node_Vr_to,   self.node_Vi_to, -(self.b)/2)
		self.stampY(Y_linear, self.node_Vi_from, self.node_Vr_from, (self.b)/2)
		self.stampY(Y_linear, self.node_Vi_to,   self.node_Vr_to, (self.b)/2)
	
	