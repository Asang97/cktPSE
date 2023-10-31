from pyomo.environ import *

def tighten_envelope(tighten_rate,vi_diff_l, vi_diff_u, vr_diff_l, vr_diff_u, B_l, B_u):
	"This is still not correct"
	print(vi_diff_l, vi_diff_u, vr_diff_l, vr_diff_u, B_l, B_u)
	vi_diff_l = tighten_rate/vi_diff_l
	vi_diff_u = tighten_rate*vi_diff_u
	vr_diff_l = tighten_rate/vr_diff_l
	vr_diff_u = tighten_rate*vr_diff_u
	B_l, B_u  = tighten_rate/B_l, tighten_rate*B_u
	print(vi_diff_l, vi_diff_u, vr_diff_l, vr_diff_u, B_l, B_u)
	return vi_diff_l, vi_diff_u, vr_diff_l, vr_diff_u, B_l, B_u
	
def add_G_noise_McCormick(model,z_RTU,num_buses,line_data,transformer,shunt,i,b,vi_diff_l,\
			vi_diff_u, vr_diff_l, vr_diff_u, B_l, B_u,vr_l,vr_u,vi_l,vi_u,nr_l,nr_u,ni_l,ni_u):
	
	model.consl.add(expr= sum(\
			   (model.v_r[i]-model.v_r[j])*line_data[i,j]['G']\
				 - model.w_i\
				 - (model.v_i[i]*0.5*line_data[i,j]['b'])\
				   for j in range(1, num_buses+1) if (i,j) in line_data.keys()and j == b)\
			  + sum(ele.calc_real_current(model.v_r[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.from_bus], model.v_i[ele.to_bus], i)\
					 for ele in transformer if ele.from_bus == i)\
					+ sum(ele.calc_real_current(model.v_r[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.from_bus], model.v_i[ele.to_bus], i)\
				  for ele in transformer if ele.to_bus == i)\
			  + sum(ele.calc_real_current(model.v_r[ele.Bus], model.v_i[ele.Bus]) for ele in shunt if ele.Bus == i)\
				   #Up there should add up to all current flow at node i except I_RTU
					 + model.v_r[i]*z_RTU[i]['p']/((z_RTU[i]['v_mag'])**2)\
					 - model.v_i[i]* z_RTU[i]['q']/((z_RTU[i]['v_mag'])**2)\
					 #This one should be I_RTU
					 + model.w_vnr[i] == 0)
					 #And the noise term
	
	model.consl.add(expr= sum(\
			   model.w_r\
				 +(model.v_i[i]-model.v_i[j])*line_data[i,j]['G']\
				 + (model.v_r[i]*0.5*line_data[i,j]['b'])\
				   for j in range(1, num_buses+1) if (i,j) in line_data.keys()and j == b)\
			  + sum(ele.calc_imag_current(model.v_r[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.from_bus], model.v_i[ele.to_bus], i)\
					 for ele in transformer if ele.from_bus == i)\
				   	+ sum(ele.calc_imag_current(model.v_r[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.from_bus], model.v_i[ele.to_bus], i)\
				  for ele in transformer if ele.to_bus == i)\
			  + sum(ele.calc_imag_current(model.v_r[ele.Bus], model.v_i[ele.Bus])\
					 for ele in shunt if ele.Bus == i)\
					 + model.v_i[i]*z_RTU[i]['p']/((z_RTU[i]['v_mag'])**2)\
					 + model.v_r[i]*z_RTU[i]['q']/((z_RTU[i]['v_mag'])**2)\
					 + model.w_vni[i] == 0)
	
	"Following are Mc_CONV for w_r = Vr_diff*B and w_i = Vi_diff*B "
	# model.consl.add(expr= sum((model.v_i[i]-model.v_i[j])*line_data[i,j]['B'] for j in range(1, num_buses+1) if (i,j) in line_data.keys()) - model.w_i + model.n_wi == 0)
	
		   
	model.consl.add(expr= sum(vi_diff_l*line_data[i,j]['B'] + (model.v_i[i]-model.v_i[j])*B_l\
							- vi_diff_l*B_l for j in range(1, num_buses+1) if (i,j) in line_data.keys()and j == b) - model.w_i <= 0)
	
	model.consl.add(expr= sum(vi_diff_u*line_data[i,j]['B'] + (model.v_i[i]-model.v_i[j])*B_u\
							- vi_diff_u*B_u for j in range(1, num_buses+1) if (i,j) in line_data.keys()and j == b) - model.w_i <= 0)
   
	model.consl.add(expr= sum(vi_diff_u*line_data[i,j]['B'] + (model.v_i[i]-model.v_i[j])*B_l\
							- vi_diff_u*B_l for j in range(1, num_buses+1) if (i,j) in line_data.keys()and j == b) - model.w_i >= 0)
	
	model.consl.add(expr= sum((model.v_i[i]-model.v_i[j])*B_u + vi_diff_l*line_data[i,j]['B']\
							- vi_diff_l*B_u for j in range(1, num_buses+1) if (i,j) in line_data.keys()and j == b) - model.w_i >= 0)
	
	model.consl.add(expr= sum((model.v_i[i]-model.v_i[j]) for j in range(1, num_buses+1) if (i,j) in line_data.keys()and j == b) <= vi_diff_u)
	
	model.consl.add(expr= sum((model.v_i[i]-model.v_i[j]) for j in range(1, num_buses+1) if (i,j) in line_data.keys()and j == b) >= vi_diff_l)
	
	# model.consl.add(expr= sum((model.v_r[i] - model.v_r[j])*line_data[i,j]['B'] for j in range(1, num_buses+1) if (i,j) in line_data.keys()) - model.w_r + model.n_wr == 0)
	
	model.consl.add(expr= sum(vr_diff_l*line_data[i,j]['B'] + (model.v_r[i]-model.v_r[j])*B_l\
							- vr_diff_l*B_l for j in range(1, num_buses+1) if (i,j) in line_data.keys()and j == b) - model.w_r <= 0)
	
	model.consl.add(expr= sum(vr_diff_u*line_data[i,j]['B'] + (model.v_r[i]-model.v_r[j])*B_u\
							- vr_diff_u*B_u for j in range(1, num_buses+1) if (i,j) in line_data.keys()and j == b) - model.w_r <= 0)
   
	model.consl.add(expr= sum(vr_diff_u*line_data[i,j]['B'] + (model.v_r[i]-model.v_r[j])*B_l\
							- vr_diff_u*B_l for j in range(1, num_buses+1) if (i,j) in line_data.keys()and j == b)- model.w_r >= 0)
	
	model.consl.add(expr= sum((model.v_r[i]-model.v_r[j])*B_u + vr_diff_l*line_data[i,j]['B']\
							- vr_diff_l*B_u for j in range(1, num_buses+1) if (i,j) in line_data.keys()and j == b)- model.w_r >= 0)
	
	model.consl.add(expr= sum((model.v_r[i]-model.v_r[j]) for j in range(1, num_buses+1) if (i,j) in line_data.keys() and j == b) <= vr_diff_u)
	
	model.consl.add(expr= sum((model.v_r[i]-model.v_r[j]) for j in range(1, num_buses+1) if (i,j) in line_data.keys()and j == b) >= vr_diff_l)
	
	model.consl.add(expr= sum(line_data[i,j]['B'] for j in range(1, num_buses+1) if (i,j) in line_data.keys() and j == b) <= B_u)
	
	model.consl.add(expr= sum(line_data[i,j]['B'] for j in range(1, num_buses+1) if (i,j) in line_data.keys() and j == b) >= B_l)
	
	"The lower and higher bound of v_r and v_i added"
	model.consl.add(expr= model.v_r[i] - vr_u <=0)
	model.consl.add(expr= model.v_r[i] - vr_l >=0)
	
	model.consl.add(expr= model.v_i[i] - vi_u <=0)
	model.consl.add(expr= model.v_i[i] - vi_l >=0)
	
	"Following are Mc_CONV for w_vnr = v_r*n_r"
	print("nr_u[i], nr_l[i], ni_u[i], ni_l[i]",nr_u[i], nr_l[i], ni_u[i], ni_l[i])
	model.consl.add(expr= model.n_r[i] - nr_u[i] <= 0)
	model.consl.add(expr= model.n_r[i] - nr_l[i] >= 0)
	
	model.consl.add(expr= model.w_vnr[i] - (nr_l[i]*model.v_r[i] + model.n_r[i]*vr_l - nr_l[i]*vr_l) >= 0)
	model.consl.add(expr= model.w_vnr[i] - (nr_u[i]*model.v_r[i] + model.n_r[i]*vr_u - nr_u[i]*vr_u) >= 0)

	model.consl.add(expr= model.w_vnr[i] - (nr_u[i]*model.v_r[i] + model.n_r[i]*vr_l - nr_u[i]*vr_l) <= 0)
	model.consl.add(expr= model.w_vnr[i] - (nr_l[i]*model.v_r[i] + model.n_r[i]*vr_u - nr_l[i]*vr_u) <= 0)


	"The lower and higher bound of v_r and v_i already added in Mc_CONV of w_r and w_n"

	"Following are Mc_CONV for w_vni = v_i*n_i"
	model.consl.add(expr= model.n_i[i] - ni_u[i] <= 0)
	model.consl.add(expr= model.n_i[i] - ni_l[i] >= 0)

	model.consl.add(expr= model.w_vni[i] - (ni_l[i]*model.v_i[i] + model.n_i[i]*vi_l - ni_l[i]*vi_l) >= 0)
	model.consl.add(expr= model.w_vni[i] - (ni_u[i]*model.v_i[i] + model.n_i[i]*vi_u - ni_u[i]*vi_u) >= 0)

	model.consl.add(expr= model.w_vni[i] - (ni_u[i]*model.v_i[i] + model.n_i[i]*vi_l - ni_u[i]*vi_l) <= 0)
	model.consl.add(expr= model.w_vni[i] - (ni_l[i]*model.v_i[i] + model.n_i[i]*vi_u - ni_l[i]*vi_u) <= 0)

	
def add_McCormick(model,z_RTU,num_buses,line_data,transformer,shunt,i,b,vi_diff_l,\
			vi_diff_u, vr_diff_l, vr_diff_u, B_l, B_u,vr_l,vr_u,vi_l,vi_u):
	
	model.consl.add(expr= sum(\
			   (model.v_r[i]-model.v_r[j])*line_data[i,j]['G']\
				 - model.w_i\
				 - (model.v_i[i]*0.5*line_data[i,j]['b'])\
				   for j in range(1, num_buses+1) if (i,j) in line_data.keys()and j == b)\
			  + sum(ele.calc_real_current(model.v_r[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.from_bus], model.v_i[ele.to_bus], i)\
					 for ele in transformer if ele.from_bus == i)\
					+ sum(ele.calc_real_current(model.v_r[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.from_bus], model.v_i[ele.to_bus], i)\
				  for ele in transformer if ele.to_bus == i)\
			  + sum(ele.calc_real_current(model.v_r[ele.Bus], model.v_i[ele.Bus]) for ele in shunt if ele.Bus == i)\
				   #Up there should add up to all current flow at node i except I_RTU
					 + model.v_r[i]*z_RTU[i]['p']/((z_RTU[i]['v_mag'])**2)\
					 - model.v_i[i]* z_RTU[i]['q']/((z_RTU[i]['v_mag'])**2)\
					 #This one should be I_RTU
					 + model.n_r[i] == 0)
					 #And the noise term
	
	model.consl.add(expr= sum(\
			   model.w_r\
				 +(model.v_i[i]-model.v_i[j])*line_data[i,j]['G']\
				 + (model.v_r[i]*0.5*line_data[i,j]['b'])\
				   for j in range(1, num_buses+1) if (i,j) in line_data.keys()and j == b)\
			  + sum(ele.calc_imag_current(model.v_r[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.from_bus], model.v_i[ele.to_bus], i) for ele in transformer if ele.from_bus == i)\
				   	+ sum(ele.calc_imag_current(model.v_r[ele.from_bus], model.v_r[ele.to_bus], model.v_i[ele.from_bus], model.v_i[ele.to_bus], i) for ele in transformer if ele.to_bus == i)\
			  + sum(ele.calc_imag_current(model.v_r[ele.Bus], model.v_i[ele.Bus]) for ele in shunt if ele.Bus == i)\
					 + model.v_i[i]*z_RTU[i]['p']/((z_RTU[i]['v_mag'])**2)\
					 + model.v_r[i]*z_RTU[i]['q']/((z_RTU[i]['v_mag'])**2)\
					 + model.n_i[i] == 0)
	
	# model.consl.add(expr= sum((model.v_i[i]-model.v_i[j])*line_data[i,j]['B'] for j in range(1, num_buses+1) if (i,j) in line_data.keys()) - model.w_i + model.n_wi == 0)
	
		   
	model.consl.add(expr= sum(vi_diff_l*line_data[i,j]['B'] + (model.v_i[i]-model.v_i[j])*B_l\
							- vi_diff_l*B_l for j in range(1, num_buses+1) if (i,j) in line_data.keys()and j == b) - model.w_i <= 0)
	
	model.consl.add(expr= sum(vi_diff_u*line_data[i,j]['B'] + (model.v_i[i]-model.v_i[j])*B_u\
							- vi_diff_u*B_u for j in range(1, num_buses+1) if (i,j) in line_data.keys()and j == b) - model.w_i <= 0)
   
	model.consl.add(expr= sum(vi_diff_u*line_data[i,j]['B'] + (model.v_i[i]-model.v_i[j])*B_l\
							- vi_diff_u*B_l for j in range(1, num_buses+1) if (i,j) in line_data.keys()and j == b) - model.w_i >= 0)
	
	model.consl.add(expr= sum((model.v_i[i]-model.v_i[j])*B_u + vi_diff_l*line_data[i,j]['B']\
							- vi_diff_l*B_u for j in range(1, num_buses+1) if (i,j) in line_data.keys()and j == b) - model.w_i >= 0)
	
	model.consl.add(expr= sum((model.v_i[i]-model.v_i[j]) for j in range(1, num_buses+1) if (i,j) in line_data.keys()and j == b) <= vi_diff_u)
	
	model.consl.add(expr= sum((model.v_i[i]-model.v_i[j]) for j in range(1, num_buses+1) if (i,j) in line_data.keys()and j == b) >= vi_diff_l)
	
	# model.consl.add(expr= sum((model.v_r[i] - model.v_r[j])*line_data[i,j]['B'] for j in range(1, num_buses+1) if (i,j) in line_data.keys()) - model.w_r + model.n_wr == 0)
	
	model.consl.add(expr= sum(vr_diff_l*line_data[i,j]['B'] + (model.v_r[i]-model.v_r[j])*B_l\
							- vr_diff_l*B_l for j in range(1, num_buses+1) if (i,j) in line_data.keys()and j == b) - model.w_r <= 0)
	
	model.consl.add(expr= sum(vr_diff_u*line_data[i,j]['B'] + (model.v_r[i]-model.v_r[j])*B_u\
							- vr_diff_u*B_u for j in range(1, num_buses+1) if (i,j) in line_data.keys()and j == b) - model.w_r <= 0)
   
	model.consl.add(expr= sum(vr_diff_u*line_data[i,j]['B'] + (model.v_r[i]-model.v_r[j])*B_l\
							- vr_diff_u*B_l for j in range(1, num_buses+1) if (i,j) in line_data.keys()and j == b)- model.w_r >= 0)
	
	model.consl.add(expr= sum((model.v_r[i]-model.v_r[j])*B_u + vr_diff_l*line_data[i,j]['B']\
							- vr_diff_l*B_u for j in range(1, num_buses+1) if (i,j) in line_data.keys()and j == b)- model.w_r >= 0)
	
	model.consl.add(expr= sum((model.v_r[i]-model.v_r[j]) for j in range(1, num_buses+1) if (i,j) in line_data.keys() and j == b) <= vr_diff_u)
	
	model.consl.add(expr= sum((model.v_r[i]-model.v_r[j]) for j in range(1, num_buses+1) if (i,j) in line_data.keys()and j == b) >= vr_diff_l)
	
	model.consl.add(expr= sum(line_data[i,j]['B'] for j in range(1, num_buses+1) if (i,j) in line_data.keys() and j == b) <= B_u)
	
	model.consl.add(expr= sum(line_data[i,j]['B'] for j in range(1, num_buses+1) if (i,j) in line_data.keys() and j == b) >= B_l)
	
	model.consl.add(expr= model.v_r[i] - vr_u <=0)
	model.consl.add(expr= model.v_r[i] - vr_l >=0)
	model.consl.add(expr= model.v_i[i] - vi_u <=0)
	model.consl.add(expr= model.v_i[i] - vi_l >=0)

def add_qc_relax(model,z_RTU,i,vr_u, vr_l, vi_u, vi_l): #TODO rewrite this to fit T_CONV
	model.consl.add(expr= model.w_vr2[i] + model.w_vi2[i] - z_RTU[i]['v_mag']**2 - model.n_v[i] == 0)
	# model.consl.add(expr= model.w_vr2[i] + model.w_vi2[i] - (z_RTU[i]['v_mag']- model.n_v[i])**2  == 0) If I can use w_vmag2[i] = (z_RTU[i]['v_mag']- model.n_v[i])**2
	
	"QC_relaxation of Vr"
	model.consl.add(expr= model.w_vr2[i] - (model.v_r[i])**2 >= 0)
	model.consl.add(expr= model.w_vr2[i] -(vr_u + vr_l)*model.v_r[i] - vr_u*vr_l <= 0)
	model.consl.add(expr= model.v_r[i] - vr_u <=0)
	model.consl.add(expr= model.v_r[i] - vr_l >=0)
	"QC_relaxation of Vi"
	model.consl.add(expr= model.w_vi2[i] - (model.v_i[i])**2 >= 0)
	model.consl.add(expr= model.w_vi2[i] -(vi_u + vi_l)*model.v_i[i] - vi_u*vi_l <= 0)
	model.consl.add(expr= model.v_i[i] - vi_u <=0)
	model.consl.add(expr= model.v_i[i] - vi_l >=0)

def add_conic_relax(model, i, z_RTU):
	model.consl.add(expr= (model.v_r[i])**2 + (model.v_i[i])**2 - z_RTU[i]['v_mag']**2 - (model.n_v[i])**2 <= 0)

# def B_l_tighten(model,z_RTU,num_buses,line_data,transformer,shunt,i,b,vi_diff_l,\
# 			vi_diff_u, vr_diff_l, vr_diff_u, B_l, B_u,vr_l,vr_u,vi_l,B_new,nc_objective):
#     B_iter = slove()
#     obj_B_min <= nc_objective
#     if (B_new - B_iter)/B_new <= tol:
#         B_new = B_iter
#     else:
#         pass
  

# def B_u_tighten(model,z_RTU,num_buses,line_data,transformer,shunt,i,b,vi_diff_l,\
# 			vi_diff_u, vr_diff_l, vr_diff_u, B_l, B_u,vr_l,vr_u,vi_l,B_new,nc_objective):
#     B_iter = slove()
#     if (B_new - B_iter)/B_new <= tol:
#       obj_B_max <= nc_objective
#       B_new = B_iter
#     else:
#         pass
#     return B_u_new