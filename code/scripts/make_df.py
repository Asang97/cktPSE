import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from scripts.options import flag_McCormick, non_convex_equality, qc_relaxation


def make_table_parameter_known(est_vr, est_vi, bus, flag_WGN):
	real_vr = []
	real_vi = []   
	real_v_mag = []
	mea_v_mag =[]
	est_v_mag = []
	for ele in bus:
		real_vr = np.append(real_vr, ele.vr_sol)
		real_vi = np.append(real_vi, ele.vi_sol)
		real_v_mag = np.append(real_v_mag, sqrt(ele.vr_sol**2+ele.vi_sol**2))
		mea_v_mag = np.append(mea_v_mag, ele.vmag_mea)
		est_v_mag = np.append(est_v_mag, ele.vmag_est)
	
	#RMSE====> NRMSE
	# rmse_vr = [sqrt(mean_squared_error(real_vr, est_vr))]
	# rmse_vi = [sqrt(mean_squared_error(real_vi, est_vi))]
	
	# rmse_vmag_real_est = [sqrt(mean_squared_error(real_v_mag, est_v_mag))]
	# rmse_vmag_real_mea = [sqrt(mean_squared_error(real_v_mag, mea_v_mag))]
	# rmse_vmag_mea_est = [sqrt(mean_squared_error(mea_v_mag, est_v_mag))]
	
	#TEST NRMSE
	rmse_vr = [sqrt(mean_squared_error(real_vr, est_vr)/real_vr^2)]
	rmse_vi = [sqrt(mean_squared_error(real_vi, est_vi)/real_vi^2)]
	
	rmse_vmag_real_est = [sqrt(mean_squared_error(real_v_mag, est_v_mag)/real_v_mag^2)]
	rmse_vmag_real_mea = [sqrt(mean_squared_error(real_v_mag, mea_v_mag)/real_v_mag^2)]
	rmse_vmag_mea_est = [sqrt(mean_squared_error(mea_v_mag, est_v_mag)/mea_v_mag^2)]
	


	table = [
		real_vr,\
		est_vr,\
		rmse_vr,\
		real_vi,\
		est_vi,\
		rmse_vi,\
		real_v_mag,\
		mea_v_mag,\
		est_v_mag,\
		rmse_vmag_real_est,\
		rmse_vmag_real_mea,\
		rmse_vmag_mea_est,\
		[flag_WGN]
	]
	
	"If giving TypeError: 'bool' object is not iterable solved for making table,"
	"print table and make sure all elements in table are list"
	
	# print(table)
	colomn_list = []
	for ele in bus:
		new_colomn = "bus"+str(ele.Bus)
		colomn_list = np.append(colomn_list, new_colomn)
	
	df = pd.DataFrame(table, columns=colomn_list,\
					  index = ['real vr','estimated vr','RMSE_vr','real vi','estimated vi','RMSE_vi'\
							   ,"real |V|","measured |V|","estimated |V|","RMSE_|V|_real_to_est", "RMSE_|V|_real_to_mea","RMSE_|V|_mea_to_est"\
							   ,'Flag_WGN'])
	df_transposed = df.transpose()
	print("==================================RESULTTABLE=================================")
	print(df_transposed.to_string())
	# print(df_transposed.to_markdown())
	print("==============================================================================")
	return (rmse_vmag_real_est,rmse_vmag_real_mea)

def make_table(est_vr, est_vi,\
			   est_B, real_B, unknown_branch, bus, flag_WGN):
	
	real_vr = []
	real_vi = []   
	real_v_mag = []
	mea_v_mag =[]
	est_v_mag = []
	
	for ele in bus:
		real_vr = np.append(real_vr, ele.vr_sol)
		real_vi = np.append(real_vi, ele.vi_sol)
		real_v_mag = np.append(real_v_mag, sqrt(ele.vr_sol**2+ele.vi_sol**2))
		mea_v_mag = np.append(mea_v_mag, ele.vmag_mea)
		est_v_mag = np.append(est_v_mag, ele.vmag_est)
	
	rmse_B = [(mean_squared_error(real_B, est_B, squared=False))]
	rmse_vr = [(mean_squared_error(real_vr, est_vr, squared=False))]
	rmse_vi = [(mean_squared_error(real_vi, est_vi, squared=False))]
	
	rmse_vmag_real_est = [(mean_squared_error(real_v_mag, est_v_mag, squared=False))]
	rmse_vmag_real_mea = [(mean_squared_error(real_v_mag, mea_v_mag, squared=False))]
	rmse_vmag_mea_est = [(mean_squared_error(mea_v_mag, est_v_mag, squared=False))]

	if flag_McCormick == False: 
		method = "Non_Convex"
	else:
		if non_convex_equality == True:
			method = "McCormick with non_convex equality"
		else:
			if qc_relaxation== True:
				method = "McCormick with QC ralaxation"
			else:
				method = "McCormick with non_convex equality removed and no QC relaxation"

	print ("Method used:",method)

	table = [
		real_vr,\
		est_vr,\
		rmse_vr,\
		real_vi,\
		est_vi,\
		rmse_vi,\
		real_v_mag,\
		mea_v_mag,\
		est_v_mag,\
		rmse_vmag_real_est,\
		rmse_vmag_real_mea,\
		rmse_vmag_mea_est,\
		# unknown_branch,\
		# real_B,\
		# est_B,\
		# rmse_B,\
		# [flag_WGN],\
		# method
	]

	table1 = [
		unknown_branch,\
		real_B,\
		est_B,\
		rmse_B,\
		[flag_WGN],
		]
	"If giving TypeError: 'bool' object is not iterable solved for making table,"
	"print table and make sure all elements in table are list"
	
	# print(table)
	colomn_list = []
	for ele in bus:
		new_colomn = "bus"+str(ele.Bus)
		colomn_list = np.append(colomn_list, new_colomn)
	
	df = pd.DataFrame(table, columns = colomn_list,\
					  index = ['real vr','estimated vr','RMSE_vr','real vi','estimated vi','RMSE_vi'\
							   ,"real |V|","measured |V|","estimated |V|","RMSE_|V|_real_to_est", "RMSE_|V|_real_to_mea","RMSE_|V|_mea_to_est"])
							#    ,"Unknown branch (i,j,ID):",'True B', 'estimated B', 'RMSE_B', 'Flag_WGN'])
	df_transposed = df.transpose()

	df1 = pd.DataFrame(table1,\
		    index = ["Unknown branch (i,j,ID):",'True B', 'estimated B', 'RMSE_B', 'Flag_WGN'])
	df1_transposed = df1.transpose()

	print("==================================================================================== STATE_VARIABLES ===================================================================================")
	print(df_transposed.to_string())
	print("=====================================================================================================================================++=================================================")
	print("+================================== PARAMETERS =================================+")
	print(df1_transposed.to_string())
	print("+=============================================================================+")
	return (rmse_vmag_real_est,rmse_vmag_real_mea)