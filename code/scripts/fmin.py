import scipy.optimize as opt
from scripts.Measure import measure_RTU

def fun(vars):
    n1, n2, n3, n4 = vars
    return n1**2 + n2**3 + n3**3 + n4**3

initialize = [0.1,0.1,0.1,0.1]
res = opt.minimize(fun,initialize)
res.x
print (res.x)