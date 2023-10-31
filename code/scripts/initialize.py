
def initialize(v_init, bus, generator, slack):
    flat_start = False
    if flat_start:
        ini_Vr = 1.0
        ini_Vi = 0.0
        ini_Q = (ele.Qmax+ele.Qmin)/2
        ini_Ir_Slack = 1e-4
        ini_Ii_Slack = 1e-4


    for ele in bus:
        v_init[ele.node_Vr] += ele.Vr_init
        v_init[ele.node_Vi] += ele.Vi_init

    for ele in generator:
        v_init[ele.node_Q] += -ele.Qinit
        # v_init[ele.node_Q] += 0.1

    for ele in slack:
        v_init[ele.node_Ir_Slack] += ele.Ir_init
        v_init[ele.node_Ii_Slack] += ele.Ii_init
        

    

    return v_init

