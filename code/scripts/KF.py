from scripts.Measure import measure_PMU
import numpy as np

def assimble_Y(v ,bus, branch, flag_WGN, num_mea, a,b):

    vs_r = []
    vs_i = []
    vt_r = []
    vt_i = []
    vs = []
    vt = []
    list_Y = {}
    
    
    for i in range (num_mea):
        temp_Y = np.zeros((2,2), dtype=complex)
        z_PMU, branch_PMU = measure_PMU(v ,bus, branch, flag_WGN)
        vs_r = np.append(vs_r,z_PMU[a]["v_r"])
        vs_i = np.append(vs_i,z_PMU[a]["v_i"])
        vt_r = np.append(vt_r,z_PMU[b]["v_r"])
        vt_i = np.append(vt_i,z_PMU[b]["v_i"])
        vs = np.append(vs, complex(z_PMU[a]["v_r"], z_PMU[a]["v_i"]))
        vt = np.append(vt, complex(z_PMU[b]["v_r"], z_PMU[b]["v_i"]))
        
        # "The index of each current is coresbounding to the matrix posiiton"
      
        index_Is = 0
        index_It = 1
        
        "observation matrix H stamping"
      
        temp_Y[index_Is,index_Is] = vs[i]
        temp_Y[index_It,index_It] = vs[i]
        temp_Y[index_It,index_Is] = vt[i]
        temp_Y[index_Is,index_It] = vt[i]
        
        "Add the H matrix to the list"
        list_Y[i] = temp_Y
    
    print("Y",list_Y)
    return list_Y


def assimble_J(v ,bus, branch, flag_WGN, num_mea, a,b):
    Is_r = []
    Is_i = []
    It_r = []
    It_i = []
    Is = []
    It =[]
    list_J = {}
    
    for i in range (num_mea):
        temp_J = np.zeros((2),dtype=complex)

        z_PMU, branch_PMU = measure_PMU(v ,bus, branch, flag_WGN)
        Is_r = np.append(Is_r,branch_PMU[(a,b)]["I_r"])
        Is_i = np.append(Is_i,branch_PMU[(a,b)]["I_i"])
        It_r = np.append(It_r,branch_PMU[(b,a)]["I_r"])
        It_i = np.append(It_i,branch_PMU[(b,a)]["I_i"])
        Is = np.append(Is, complex(branch_PMU[(a,b)]["I_r"],branch_PMU[(a,b)]["I_i"]))
        It = np.append(It, complex(branch_PMU[(b,a)]["I_r"],branch_PMU[(b,a)]["I_i"]))
        
        "The index of each current is coresbounding to the matrix posiiton"

        index_Is = 0
        index_It = 1
        
        "J matrix stamping"
        temp_J[index_Is] += Is[i]
        temp_J[index_It] += It[i]

        list_J[i] = temp_J


    print("J",list_J)
    return list_J
