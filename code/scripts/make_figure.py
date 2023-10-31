import matplotlib.pyplot as plt
import numpy as np
from pyomo.environ import value
from math import sqrt

def initialize_list():
	list_z_vmag = []
	list_real_vmag = []
	list_est_vmag = []
	return list_z_vmag, list_real_vmag, list_est_vmag
    
def storage_data(list_z_vmag, list_real_vmag, list_est_vmag, bus_ele):
	# est_vr  = bus_ele.vr_est
	# est_vi  = bus_ele.vi_est
	# real_vr = bus_ele.vr_sol
	# real_vi = bus_ele.vi_sol
	mea_vmag= bus_ele.vmag_mea
	real_vmag = sqrt(bus_ele.vr_sol**2+bus_ele.vi_sol**2)
	est_vmag = bus_ele.vmag_est

	list_z_vmag = np.append(list_z_vmag, mea_vmag)
	list_real_vmag = np.append(list_real_vmag, real_vmag)
	list_est_vmag = np.append(list_est_vmag, est_vmag)
	# list_est_vr = np.append(list_est_vr,est_vr)
	# list_est_vi = np.append(list_est_vi,est_vi)
	# list_real_vr = np.append(list_real_vr,real_vr)
	# list_real_vi = np.append(list_real_vi,real_vi)
	
	return list_z_vmag, list_real_vmag, list_est_vmag


def make_plot(list_mea,list_real,list_est):
	plt.plot(list_real)
	plt.plot(list_mea,color='red')
	plt.plot(list_est, color='green')
	plt.show()
	
