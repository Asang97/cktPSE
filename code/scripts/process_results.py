import math as mt
import numpy as np

# def process_results(v,bus, branch):
#     print ("===============================================================================================")
#     print ("      Bus Data")
#     print ("===============================================================================================")
#     voltage_list = np.array([])
#     ang_list = np.array([])
#     for ele in bus:
        

#         if ele.Type == 1 or ele.Type == 3:
#             # test = (v[ele.node_Vi])/(v[ele.node_Vr])
#             ang = np.rad2deg(mt.atan(v[ele.node_Vi])/(v[ele.node_Vr]))
#             voltage_list =np.append(voltage_list, (v[ele.node_Vr]**2 + v[ele.node_Vi]**2)**(1/2))
#             ang_list = np.append(ang_list, ang)
#             print ("The Voltage magnitude |V| at bus",ele.Bus,"is",(v[ele.node_Vr]**2 + v[ele.node_Vi]**2)**(1/2), "with angle of", ang)
#         if ele.Type ==2:
#             ang = np.rad2deg(mt.atan(v[ele.node_Vi])/(v[ele.node_Vr]))
#             voltage_list =np.append(voltage_list, (v[ele.node_Vr]**2 + v[ele.node_Vi]**2)**(1/2))
#             ang_list = np.append(ang_list, ang)
#             print ("The Voltage magnitude |V| at bus",ele.Bus,"is",(v[ele.node_Vr]**2 + v[ele.node_Vi]**2)**(1/2),"with angle of", ang, "The power Q generated is", -v[ele.node_Q]*100, "MVA")
    
#     print ('maximum magitude', np.max(voltage_list),"minimum maginitude", np.min(voltage_list),'maximum ang', np.max(ang_list),"minimum ang", np.min(ang_list))
#     print ("===============================================================================================")
#     print ("      Branch Data")
#     print ("===============================================================================================")
#     for ele in branch:
#         if ele.from_bus == 68 and ele.to_bus == 116:
#             print (ele.B_l)
#             pass

def process_results(v, bus, slack):
    for ele in slack:
        print ("SLACK AT",ele.Bus)
    print ("===============================================================================================")
    print ("      Bus Data")
    print ("===============================================================================================")
    for ele in bus:
        # PQ bus
        if ele.Type == 1:
            Vr = v[ele.node_Vr]
            Vi = v[ele.node_Vi]
            Vmag = np.sqrt(Vr**2 + Vi**2)
            Vth = np.arctan2(Vi, Vr) * 180/np.pi
            print("%d, Vmag: %.3f p.u.,Vr: %.3f,Vi: %.3f, Vth: %.3f deg" % (ele.Bus, Vmag, Vr, Vi, Vth))
        # PV bus
        elif ele.Type == 2:
            Vr = v[ele.node_Vr]
            Vi = v[ele.node_Vi]
            Vmag = np.sqrt(Vr**2 + Vi**2)
            Vth = np.arctan2(Vi, Vr) * 180.0/np.pi
            Qg = v[ele.node_Q]*100
            print("%d: Vmag: %.3f p.u., Vr: %.3f,Vi: %.3f, Vth: %.3f deg, Qg: %.3f MVAr" % (ele.Bus, Vmag, Vr, Vi, Vth, Qg))
        elif ele.Type == 3:
            Vr = v[ele.node_Vr]
            Vi = v[ele.node_Vi]
            Vmag = np.sqrt(Vr**2 + Vi**2)
            Vth = np.arctan2(Vi, Vr) * 180/np.pi
            Pg, Qg = slack[0].calc_slack_PQ(v)
            print("SLACK: %d, Vmag: %.3f p.u., Vth: %.3f deg, Pg: %.3f MW, Qg: %.3f MVar" % (ele.Bus, Vmag, Vth, Pg*100, Qg*100))
