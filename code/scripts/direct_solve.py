from scripts.Measure import measure_RTU
import numpy as np

def stamp_J(size_Y):
    J_matrix_dc = np.ones(size_Y)*1e-5
    return J_matrix_dc

def stamp_Y(bus, branch, size_Y):
    Y_matrix_dc = np.zeros([size_Y,size_Y])
    for ele in bus:
        'Get singular, need to remove slack bus'
        Y_matrix_dc[ele.node_Vr, ele.node_Vr] +=  ele.g_rtu
        Y_matrix_dc[ele.node_Vi, ele.node_Vi] +=  ele.g_rtu
        Y_matrix_dc[ele.node_Vr, ele.node_Vi] += -ele.b_rtu
        Y_matrix_dc[ele.node_Vi, ele.node_Vr] +=  ele.b_rtu
        
    for ele in branch:
         
         ele.stamp(Y_matrix_dc)
    # print (Y_matrix_dc)
    return Y_matrix_dc

def direct_solve(y,j):
    return(np.linalg.solve(y,j))

"Solve linear function with matrix regardless of the noise"
"Generate this just to compare with the estimated results"