from __future__ import division
from itertools import count
from models.Buses import Buses
from scipy.sparse import csr_matrix
import numpy as np
from typing import Union, Optional
from pyomo.environ import Var as PyomoVar

class Loads:
    _ids = count(0)

    def __init__(self,
                 Bus,
                 P,
                 Q,
                 IP,
                 IQ,
                 ZP,
                 ZQ,
                 area,
                 status):
        """Initialize an instance of a PQ or ZIP load in the power grid.

        Args:
            Bus (int): the bus where the load is located
            P (float): the active power of a constant power (PQ) load.
            Q (float): the reactive power of a constant power (PQ) load.
            IP (float): the active power component of a constant current load.
            IQ (float): the reactive power component of a constant current load.
            ZP (float): the active power component of a constant admittance load.
            ZQ (float): the reactive power component of a constant admittance load.
            area (int): location where the load is assigned to.
            status (bool): indicates if the load is in-service or out-of-service.
        """
        self.id = Loads._ids.__next__()
        self.Bus = Bus
        
        # Normalize P and Q to convert into pu system
        self.P = P/100
        self.Q = Q/100
        
        # two orgin values added to regain the P and Q after the last loop of load factor
        self.P_origin = P/100
        self.Q_origin = Q/100 

        self.row_list    = np.array([])
        self.colom_list  = np.array([])
        self.value_list  = np.array([])
       
        # You will need to implement the remainder of the __init__ function yourself.
        # You should also add some other class functions you deem necessary for stamping,
        # initializing, and processing results.


    def assign_idx(self, bus):
        self.node_Vr = bus[Buses.all_bus_key_[self.Bus]].node_Vr
        self.node_Vi = bus[Buses.all_bus_key_[self.Bus]].node_Vi

    def stampY(self, Y_matrix, row, colom, value):
        Y_matrix[row, colom] += value

    def stampJ(self, J_matrix, colom, value):
        J_matrix[colom] += value

    def assimble_sparse_stampY(self, Y_sparse_matrix, row_list, colom_list, value_list, size_Y):
        Y_sparse_matrix += csr_matrix ((value_list, (row_list, colom_list)), shape=(size_Y, size_Y))
         # note the value, row, colom should be np.array form
    
    def assimble_sparse_stampJ(self, J_sparse_matrix, row_list, colom_list, value_list, size_Y):
        J_sparse_matrix += csr_matrix ((value_list, (row_list, colom_list)), shape=(size_Y, 1))
    
    
    def sparse_stampY(self, row, colom, value):
        
        self.row_list   = np.append(self.row_list,row)
        self.colom_list = np.append(self.colom_list,colom)
        self.value_list = np.append(self.value_list,value)

    # def sparse_stampJ(self, row_list, colom_list, value_list, row, value):
    #     row_list   = np.append(row_list,row)
    #     colom_list = np.append(colom_list,0.0)
    #     value_list = np.append(value_list,value)

    def sparse_stamp_assimbleY(self, v):

        self.row_list    = np.array([])
        self.colom_list  = np.array([])
        self.value_list  = np.array([])
        
        P_l   = self.P
        Q_l   = self.Q
        V_rl  = v[self.node_Vr]
        V_il  = v[self.node_Vi]

       # Real part
        I_rl_hist  = (P_l*V_rl + Q_l*V_il)/(V_rl**2 + V_il**2)
        dI_rldV_rl = (P_l*(V_il**2 - V_rl**2) - 2*Q_l*V_rl*V_il)/(V_rl**2 + V_il**2)**2
        dI_rldV_il = (Q_l*(V_rl**2 - V_il**2) - 2*P_l*V_rl*V_il)/(V_rl**2 + V_il**2)**2

        # fr_hist at J matrix
        fr_hist = - I_rl_hist + dI_rldV_rl*V_rl + dI_rldV_il*V_il

        # Imaginary part
        I_il_hist  = (P_l*V_il - Q_l*V_rl)/(V_rl**2 + V_il**2)
        dI_ildV_rl = (Q_l*(V_rl**2 - V_il**2) - 2*P_l*V_rl*V_il)/(V_rl**2 + V_il**2)**2
        dI_ildV_il = (P_l*(V_rl**2 - V_il**2) + 2*Q_l*V_rl*V_il)/(V_rl**2 + V_il**2)**2
        
        # fi_hist at J matrix
        fi_hist = - I_il_hist + dI_ildV_rl*V_rl + dI_ildV_il*V_il
        
        self.sparse_stampY(self.node_Vr,self.node_Vr,dI_rldV_rl)
        self.sparse_stampY(self.node_Vr,self.node_Vi,dI_rldV_il)
        self.sparse_stampY(self.node_Vi,self.node_Vr,dI_ildV_rl)
        self.sparse_stampY(self.node_Vi,self.node_Vi,dI_ildV_il)

        return self.row_list, self.colom_list, self.value_list


    def stamp(self, v, Y_nonlinear, J_nonliniear):
        P_l   = self.P
        Q_l   = self.Q
        V_rl  = v[self.node_Vr]
        V_il  = v[self.node_Vi]

        # Real part
        I_rl_hist  = (P_l*V_rl + Q_l*V_il)/(V_rl**2 + V_il**2)
        dI_rldV_rl = (P_l*(V_il**2 - V_rl**2) - 2*Q_l*V_rl*V_il)/(V_rl**2 + V_il**2)**2
        dI_rldV_il = (Q_l*(V_rl**2 - V_il**2) - 2*P_l*V_rl*V_il)/(V_rl**2 + V_il**2)**2

        # fr_hist at J matrix
        fr_hist = - I_rl_hist + dI_rldV_rl*V_rl + dI_rldV_il*V_il

        # Imaginary part
        I_il_hist  = (P_l*V_il - Q_l*V_rl)/(V_rl**2 + V_il**2)
        dI_ildV_rl = (Q_l*(V_rl**2 - V_il**2) - 2*P_l*V_rl*V_il)/(V_rl**2 + V_il**2)**2
        dI_ildV_il = (P_l*(V_rl**2 - V_il**2) + 2*Q_l*V_rl*V_il)/(V_rl**2 + V_il**2)**2
        
        # fi_hist at J matrix
        fi_hist = - I_il_hist + dI_ildV_rl*V_rl + dI_ildV_il*V_il

        self.stampY(Y_nonlinear,self.node_Vr,self.node_Vr,dI_rldV_rl)
        self.stampY(Y_nonlinear,self.node_Vr,self.node_Vi,dI_rldV_il)
        self.stampY(Y_nonlinear,self.node_Vi,self.node_Vr,dI_ildV_rl)
        self.stampY(Y_nonlinear,self.node_Vi,self.node_Vi,dI_ildV_il)
        self.stampJ(J_nonliniear,self.node_Vr, fr_hist)
        self.stampJ(J_nonliniear,self.node_Vi, fi_hist)

    def stamp_J(self, v, J_nonliniear):
        P_l   = self.P
        Q_l   = self.Q
        V_rl  = v[self.node_Vr]
        V_il  = v[self.node_Vi]

        # Real part
        I_rl_hist  = (P_l*V_rl + Q_l*V_il)/(V_rl**2 + V_il**2)
        dI_rldV_rl = (P_l*(V_il**2 - V_rl**2) - 2*Q_l*V_rl*V_il)/(V_rl**2 + V_il**2)**2
        dI_rldV_il = (Q_l*(V_rl**2 - V_il**2) - 2*P_l*V_rl*V_il)/(V_rl**2 + V_il**2)**2

        # fr_hist at J matrix
        fr_hist = - I_rl_hist + dI_rldV_rl*V_rl + dI_rldV_il*V_il

        # Imaginary part
        I_il_hist  = (P_l*V_il - Q_l*V_rl)/(V_rl**2 + V_il**2)
        dI_ildV_rl = (Q_l*(V_rl**2 - V_il**2) - 2*P_l*V_rl*V_il)/(V_rl**2 + V_il**2)**2
        dI_ildV_il = (P_l*(V_rl**2 - V_il**2) + 2*Q_l*V_rl*V_il)/(V_rl**2 + V_il**2)**2
        
        # fi_hist at J matrix
        fi_hist = - I_il_hist + dI_ildV_rl*V_rl + dI_ildV_il*V_il

        
        self.stampJ(J_nonliniear,self.node_Vr, fr_hist)
        self.stampJ(J_nonliniear,self.node_Vi, fi_hist)

    def calc_Ir(self, 
                Vr: Union[PyomoVar, float],
                Vi: Union[PyomoVar, float],
                p:  Union[PyomoVar, float],
                q:  Union[PyomoVar, float],
                ):
        Ir_load = (p* Vr + q * Vi) / (Vr ** 2 + Vi ** 2)
        return Ir_load
    
    def calc_Ii(self, 
                Vr: Union[PyomoVar, float],
                Vi: Union[PyomoVar, float],
                p:  Union[PyomoVar, float],
                q:  Union[PyomoVar, float],
                ):
        Ii_load = (p * Vi - q*Vr) / (Vr ** 2 + Vi ** 2)
        return Ii_load

    
    



        
