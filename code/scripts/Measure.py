import numpy as np
import math as mt
from models.Buses import Buses
from models.Rtu import Rtu
from scripts.options import same_noise, random_seed, distribution
dist = distribution

# if get_new_seed:
#        new_test_wgn_list = np.random.normal(0,dist,3)
#        test_wgn_list = np.copy(new_test_wgn_list)
# else:
#        pass

# test_wgn_list = [-1.74935829e-03, -5.45280485e-04, -6.89934198e-04]
# print (test_wgn_list[0])

def make_noisy(z,dist,random_seed): 
	   # TODO Just a thought, you can use random(seed) with a certain seed to create a replicable list of random seed, and use them to generate random noises 
	if same_noise:
		np.random.seed(random_seed)
		test_wgn = np.random.normal(z,dist,1)
		return test_wgn[0]
	else:
		wgn = np.random.normal(z,dist,1)
		return wgn[0]

def measure_RTU(v, bus, branch, transformer, shunt, flag_WGN):
	#TODO Cope input, this part needs to be rewrite
	z_RTU = {}
	z_RTU0 = {}
	# z_special = {}
	answer = {}
	direct_calculation = {}

	add_WGN = flag_WGN
	test_FT = False
	   # if need WGN added, swith on
	
	intergrated_RTU_list = []
	
	for buses in bus:
		
		ang = np.rad2deg(mt.atan(v[buses.node_Vi])/(v[buses.node_Vr]))
		v_r = v[buses.node_Vr]
		v_i = v[buses.node_Vi]
		
		if add_WGN:
			v_mag0 = (v_r**2 + v_i**2)**(1/2)
			v_mag = make_noisy((v_r**2 + v_i**2)**(1/2), dist, random_seed) 
		else:
			v_mag0 = (v_r**2 + v_i**2)**(1/2)
			v_mag = (v_r**2 + v_i**2)**(1/2)
		
		z_RTU[buses.Bus] = {'v_mag':v_mag}
		answer[buses.Bus] = {'v_mag':v_mag, 'Ang':ang, 'Vr':v_r, 'Vi':v_i}
		
		line_data = {}
		
		for branches in branch:
			line_data[branches.from_bus, branches.to_bus] = {'G': branches.G_l, 'B': branches.B_l, 'b': branches.b}
			line_data[branches.to_bus, branches.from_bus] = {'G': branches.G_l, 'B': branches.B_l, 'b': branches.b}
		
		transformer_data = {}
		
		for ele in transformer:
			transformer_data[ele.from_bus, ele.to_bus] = {'G':ele.G_l, 'B':ele.B_l,'tr':ele.tr, 'ang':ele.ang}

		#TODO fix the currents, these are off except the Bus1 I_r
		I_r_line, I_i_line = 0.0, 0.0
		
		# calc sum of branch current at bus i 
		for (i,j) in line_data.keys():
			if i == buses.Bus:
				#TODO also replace this to Class branch
				I_r_line += -((v[bus[Buses.all_bus_key_[i]].node_Vr]-v[bus[Buses.all_bus_key_[j]].node_Vr])*line_data[i,j]['G']\
									- (v[bus[Buses.all_bus_key_[i]].node_Vi]-v[bus[Buses.all_bus_key_[j]].node_Vi])*(line_data[i,j]['B'])\
										- (v[bus[Buses.all_bus_key_[i]].node_Vi]*0.5*line_data[i,j]['b']))

				I_i_line += -((v[bus[Buses.all_bus_key_[i]].node_Vr]-v[bus[Buses.all_bus_key_[j]].node_Vr])*(line_data[i,j]['B'])\
									+ (v[bus[Buses.all_bus_key_[i]].node_Vi]-v[bus[Buses.all_bus_key_[j]].node_Vi])*(line_data[i,j]['G'])\
									+ (v[bus[Buses.all_bus_key_[i]].node_Vr]*0.5*line_data[i,j]['b']))
				
				# print((i,j), I_r, I_i)
		
		# calc sum of transformer current at bus i 
		I_r_transformer = -(sum(ele.calc_real_current_measure(v[bus[Buses.all_bus_key_[ele.from_bus]].node_Vr], v[bus[Buses.all_bus_key_[ele.to_bus]].node_Vr], v[bus[Buses.all_bus_key_[ele.from_bus]].node_Vi], v[bus[Buses.all_bus_key_[ele.to_bus]].node_Vi], buses.Bus) for ele in transformer if ele.from_bus == buses.Bus)\
			+ sum(ele.calc_real_current_measure(v[bus[Buses.all_bus_key_[ele.from_bus]].node_Vr], v[bus[Buses.all_bus_key_[ele.to_bus]].node_Vr], v[bus[Buses.all_bus_key_[ele.from_bus]].node_Vi], v[bus[Buses.all_bus_key_[ele.to_bus]].node_Vi], buses.Bus) for ele in transformer if ele.to_bus == buses.Bus))
		I_i_trasnformer = -(sum(ele.calc_imag_current_measure(v[bus[Buses.all_bus_key_[ele.from_bus]].node_Vr], v[bus[Buses.all_bus_key_[ele.to_bus]].node_Vr], v[bus[Buses.all_bus_key_[ele.from_bus]].node_Vi], v[bus[Buses.all_bus_key_[ele.to_bus]].node_Vi], buses.Bus) for ele in transformer if ele.from_bus == buses.Bus)\
			+ sum(ele.calc_imag_current_measure(v[bus[Buses.all_bus_key_[ele.from_bus]].node_Vr], v[bus[Buses.all_bus_key_[ele.to_bus]].node_Vr], v[bus[Buses.all_bus_key_[ele.from_bus]].node_Vi], v[bus[Buses.all_bus_key_[ele.to_bus]].node_Vi], buses.Bus) for ele in transformer if ele.to_bus == buses.Bus))
		
		# calc sum of shunt current at bus i
		I_r_shunt = -(sum(ele.calc_real_current_measure(v[bus[Buses.all_bus_key_[ele.Bus]].node_Vr], v[bus[Buses.all_bus_key_[ele.Bus]].node_Vi])for ele in shunt if ele.Bus == buses.Bus))
		I_i_shunt = -(sum(ele.calc_imag_current_measure(v[bus[Buses.all_bus_key_[ele.Bus]].node_Vr], v[bus[Buses.all_bus_key_[ele.Bus]].node_Vi])for ele in shunt if ele.Bus == buses.Bus))

		# calc sum of current at bus i 
		I_r = I_r_line + I_r_transformer + I_r_shunt
		I_i = I_i_line + I_i_trasnformer + I_i_shunt
		
		
		if add_WGN:
			p = v_r*I_r + v_i*I_i
			q = -(v_r*I_i - v_i*I_r)
			p0 = v_r*I_r + v_i*I_i
			q0 = -(v_r*I_i - v_i*I_r)
			p = make_noisy(p,dist, random_seed)
			q = make_noisy(q,dist, random_seed)
		
			# print("Noise_added", "P:",p0-p,"Q:", q0-q)
		
		if test_FT:
			p = v_r*I_r + v_i*I_i
			q = -(v_r*I_i - v_i*I_r)
			p = p + make_noisy(0,dist,random_seed)
			q = q + make_noisy(0,dist,random_seed)   
		else:
			p0 = v_r*I_r + v_i*I_i
			q0 = -(v_r*I_i - v_i*I_r)
			p = v_r*I_r + v_i*I_i
			q = -(v_r*I_i - v_i*I_r)
				
		# Store data into z_RTU and z_RTU0
		z_RTU[buses.Bus] = {'v_mag':v_mag, 'p':p, 'q':-q}
		z_RTU0[buses.Bus] = {'v_mag':v_mag0, 'p':p0, 'q':-q0, 'vr':v_r,'vi':v_i, 'Ir':I_r, 'Ii':I_i,'Ir_line':I_r_line, 'Ii_line':I_i_line,'Ir_trans':I_r_transformer, 'Ii_trans':I_i_trasnformer,'Ir_shunt':I_r_shunt, 'Ii_shunt':I_i_shunt }
		
		# Realization for each RTU, take ZI nodes out
		if abs(p0) <= 1e-6 and abs(q0) <= 1e-6:
			pass
		else:
			r = Rtu(buses.Bus, buses.Type, v_mag, p, -q,flag_WGN)
			intergrated_RTU_list.append(r)
		
		# Get direct_calc results (NOT USED)
		# s_mag = mt.sqrt(p**2+q**2)
		# print(s_mag)
		# theta =  mt.acos(p/s_mag)
		# v_r_dc = v_mag*(mt.cos(theta))
		# v_i_dc = v_mag*(mt.sin(theta))
		# direct_calculation[buses.Bus] = {"v_r_dc":v_r_dc,"v_i_dc":v_i_dc}
	
	return z_RTU0, z_RTU, intergrated_RTU_list #Z_RTU0 is noiseless results


def measure_PMU(v, bus, branch, flag_WGN):
	z_PMU = {}
	branch_PMU={}
	answer = {}
	dist = 0.001
	add_WGN = flag_WGN
	# if need WGN added, swith on

	for buses in bus:
		ang = np.rad2deg(mt.atan(v[buses.node_Vi])/(v[buses.node_Vr]))
		v_r = v[buses.node_Vr]
		v_i = v[buses.node_Vi]
		if add_WGN:
			v_mag = make_noisy((v_r**2 + v_i**2)**(1/2), dist)
		else:
			v_mag = (v_r**2 + v_i**2)**(1/2)
		
		answer[buses.Bus] = {'v_mag':v_mag, 'Ang':ang, 'Vr':v_r, 'Vi':v_i}
		line_data = {}

		for branches in branch:
			line_data[branches.from_bus, branches.to_bus] = {'G': branches.G_l, 'B': branches.B_l, 'b': branches.b}
			line_data[branches.to_bus, branches.from_bus] = {'G': branches.G_l, 'B': branches.B_l, 'b': branches.b}


		#  print(line_data.keys())

		#TODO fix the currents, these are off except the Bus1 I_r
		I_r, I_i = 0.0, 0.0
		for (i,j) in line_data.keys():
			if i == buses.Bus:
				I_r += -((v[bus[Buses.all_bus_key_[i]].node_Vr]-v[bus[Buses.all_bus_key_[j]].node_Vr])*line_data[i,j]['G']\
									- (v[bus[Buses.all_bus_key_[i]].node_Vi]-v[bus[Buses.all_bus_key_[j]].node_Vi])*(line_data[i,j]['B'])\
										- (v[bus[Buses.all_bus_key_[i]].node_Vi]*0.5*line_data[i,j]['b']))

				I_i += -((v[bus[Buses.all_bus_key_[i]].node_Vr]-v[bus[Buses.all_bus_key_[j]].node_Vr])*(line_data[i,j]['B'])\
									+ (v[bus[Buses.all_bus_key_[i]].node_Vi]-v[bus[Buses.all_bus_key_[j]].node_Vi])*(line_data[i,j]['G'])\
									+ (v[bus[Buses.all_bus_key_[i]].node_Vr]*0.5*line_data[i,j]['b']))
				# print((i,j), I_r, I_i)

		for (i,j) in line_data.keys():
			if i == buses.Bus:
				I_r = -((v[bus[Buses.all_bus_key_[i]].node_Vr]-v[bus[Buses.all_bus_key_[j]].node_Vr])*line_data[i,j]['G']\
									- (v[bus[Buses.all_bus_key_[i]].node_Vi]-v[bus[Buses.all_bus_key_[j]].node_Vi])*(line_data[i,j]['B'])\
										- (v[bus[Buses.all_bus_key_[i]].node_Vi]*0.5*line_data[i,j]['b']))

				I_i = -((v[bus[Buses.all_bus_key_[i]].node_Vr]-v[bus[Buses.all_bus_key_[j]].node_Vr])*(line_data[i,j]['B'])\
									+ (v[bus[Buses.all_bus_key_[i]].node_Vi]-v[bus[Buses.all_bus_key_[j]].node_Vi])*(line_data[i,j]['G'])\
									+ (v[bus[Buses.all_bus_key_[i]].node_Vr]*0.5*line_data[i,j]['b']))
				# print((i,j), I_r, I_i)
				if add_WGN:
						I_r = make_noisy(I_r,dist)
						I_i = make_noisy(I_i,dist)
						branch_PMU[(i,j)] = {"I_r":I_r, "I_i":I_i}
				else:
						branch_PMU[(i,j)] = {"I_r":I_r, "I_i":I_i}


		if add_WGN:
			v_r_PMU = make_noisy(v_r,dist)
			v_i_PMU = make_noisy(v_i,dist)                     
			I_r_PMU = make_noisy(I_r,dist)                    
			I_i_PMU = make_noisy(I_i,dist)
		else:    
			v_r_PMU = v_r
			v_i_PMU = v_i
			I_r_PMU = I_r
			I_i_PMU = I_i
		
		z_PMU[buses.Bus] = {'v_r':v_r_PMU, 'v_i':v_i_PMU, 'I_r': I_r_PMU, 'I_i':I_i_PMU}

	return z_PMU, branch_PMU

# Run measure again with