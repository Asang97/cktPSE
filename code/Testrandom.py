
import numpy as np
import math as mt
from run_solver import run
same_noise = False

def make_noisy(z,dist,random_seed):
       if same_noise:
              np.random.seed(random_seed)
              test_wgn = np.random.normal(z,dist,1)
              return test_wgn[0]
       else:
              wgn = np.random.normal(z,dist,1)
              return wgn[0]
       
print (make_noisy(0,0,1))
print (make_noisy(2,0,1))
print (make_noisy(3,0,1))
print (make_noisy(4,0,1))
print (make_noisy(5,0,1))
