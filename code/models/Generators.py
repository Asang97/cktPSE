from __future__ import division
from itertools import count
from scripts.global_vars import global_vars
from models.Buses import Buses
from scipy.sparse import csr_matrix
import numpy as np
from typing import Union, Optional
from pyomo.environ import Var as PyomoVar

class Generators:
    _ids = count(0)
    RemoteBusGens = dict()
    RemoteBusRMPCT = dict()
    gen_bus_key_ = {}
    total_P = 0

    def __init__(self,
                 Bus,
                 P,
                 Vset,
                 Qmax,
                 Qmin,
                 Pmax,
                 Pmin,
                 Qinit,
                 RemoteBus,
                 RMPCT,
                 gen_type):
        """Initialize an instance of a generator in the power grid.

        Args:
            Bus (int): the bus number where the generator is located.
            P (float): the current amount of active power the generator is providing.
            Vset (float): the voltage setpoint that the generator must remain fixed at.
            Qmax (float): maximum reactive power
            Qmin (float): minimum reactive power
            Pmax (float): maximum active power
            Pmin (float): minimum active power
            Qinit (float): the initial amount of reactive power that the generator is supplying or absorbing.
            RemoteBus (int): the remote bus that the generator is controlling
            RMPCT (float): the percent of total MVAR required to hand the voltage at the controlled bus
            gen_type (str): the type of generator
        """

        self.id = self._ids.__next__()
        self.Bus = Bus
        self.Qmax_MVAR = Qmax
        self.Qmin_MVAR = Qmin
        self.Pmax_MW = Pmax
        self.Pmin_MW = Pmin
        self.Qinit_MVAR = Qinit
        self.RemoteBus = RemoteBus
        self.RMPCT = RMPCT
        self.gen_type = gen_type
        #Normalize the P with 100 to convert to pu
        self.P = P/100
        
        # origin value is added to regain the P and Q after the last loop of load factor
        self.P_origin = P/100
        
        self.Vset = Vset
        self.Qmax = Qmax/100
        self.Qmin = Qmin/100
        self.Qinit = Qinit/100
        self.Pmax = Pmax/100
        self.Pmin = Pmin/100
        
        
        self.row_list    = np.array([])
        self.colom_list  = np.array([])
        self.value_list  = np.array([])
        

    def assign_idx(self, bus):
        
        self.node_Vr = bus[Buses.all_bus_key_[self.Bus]].node_Vr
        self.node_Vi = bus[Buses.all_bus_key_[self.Bus]].node_Vi
        self.node_Q  = bus[Buses.all_bus_key_[self.Bus]].node_Q
    
    def stampY(self, Y_matrix, row, colom, value):
        Y_matrix[row, colom] += value

    def stampJ(self, J_matrix, colom, value):
        J_matrix[colom] += value

    # def assimble_sparse_stampY(self, Y_sparse_matrix, row_list, colom_list, value_list, size_Y):
    #     Y_sparse_matrix += csr_matrix ((value_list, (row_list, colom_list)), shape=(size_Y, size_Y))
    #      # note the value, row, colom should be np.array form
    
    # def assimble_sparse_stampJ(self, J_sparse_matrix, row_list, colom_list, value_list, size_Y):
    #     J_sparse_matrix += csr_matrix ((value_list, (row_list, colom_list)), shape=(size_Y, 1))
    
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

        P_g   = self.P
        V_rg  = v[self.node_Vr]
        V_ig  = v[self.node_Vi]
        Q_g   = v[self.node_Q]

        # Real part
        I_rg_hist  = (-P_g*V_rg - Q_g*V_ig)/(V_rg**2 + V_ig**2)
        
        dI_rgdV_rg = (P_g*(V_rg**2 - V_ig**2) + 2*Q_g*V_rg*V_ig)/(V_rg**2 + V_ig**2)**2
        dI_rgdV_ig = (Q_g*(V_ig**2 - V_rg**2) + 2*P_g*V_rg*V_ig)/(V_rg**2 + V_ig**2)**2
        
        dI_rgdQ_g  = -V_ig/(V_rg**2 + V_ig**2)

        # fr_hist at J matrix
        fr_hist = - I_rg_hist + dI_rgdV_rg*V_rg + dI_rgdV_ig*V_ig + dI_rgdQ_g*Q_g

        # Imaginarry part
        I_ig_hist = (-P_g*V_ig + Q_g*V_rg)/(V_rg**2 + V_ig**2)

        dI_igdV_rg = (Q_g*(V_ig**2 - V_rg**2) + 2*P_g*V_rg*V_ig)/(V_rg**2 + V_ig**2)**2
        dI_igdV_ig = (P_g*(V_ig**2 - V_rg**2) - 2*Q_g*V_rg*V_ig)/(V_rg**2 + V_ig**2)**2
        
        dI_igdQ_g  = V_rg/(V_rg**2 + V_ig**2)

        # fi_hist at J matrix
        fi_hist = - I_ig_hist + dI_igdV_rg*V_rg + dI_igdV_ig*V_ig + dI_igdQ_g*Q_g

        # Veq_hist at J matrix
        Veq_hist = V_rg**2 + V_ig**2 + (self.Vset)**2


        # 4 partial of I and V
        self.sparse_stampY( self.node_Vr, self.node_Vr, dI_rgdV_rg)
        self.sparse_stampY( self.node_Vr, self.node_Vi, dI_rgdV_ig)
        self.sparse_stampY( self.node_Vi, self.node_Vr, dI_igdV_rg)
        self.sparse_stampY( self.node_Vi, self.node_Vi, dI_igdV_ig)

        # 2 partial of I and Q
        self.sparse_stampY( self.node_Vr, self.node_Q, dI_rgdQ_g)
        self.sparse_stampY( self.node_Vi, self.node_Q, dI_igdQ_g)

        # 2 partial of Vrg and Vig
        self.sparse_stampY( self.node_Q, self.node_Vr, 2*V_rg)
        self.sparse_stampY( self.node_Q, self.node_Vi, 2*V_ig)
        
        return self.row_list, self.colom_list, self.value_list
    
    def stamp(self, v, Y_nonlinear, J_nonliniear):
        P_g   = self.P
        V_rg  = v[self.node_Vr]
        V_ig  = v[self.node_Vi]
        Q_g   = v[self.node_Q]

        # Real part
        I_rg_hist  = (-P_g*V_rg - Q_g*V_ig)/(V_rg**2 + V_ig**2)
        
        dI_rgdV_rg = (P_g*(V_rg**2 - V_ig**2) + 2*Q_g*V_rg*V_ig)/(V_rg**2 + V_ig**2)**2
        dI_rgdV_ig = (Q_g*(V_ig**2 - V_rg**2) + 2*P_g*V_rg*V_ig)/(V_rg**2 + V_ig**2)**2
        
        dI_rgdQ_g  = -V_ig/(V_rg**2 + V_ig**2)

        # fr_hist at J matrix
        fr_hist = - I_rg_hist + dI_rgdV_rg*V_rg + dI_rgdV_ig*V_ig + dI_rgdQ_g*Q_g

        # Imaginarry part
        I_ig_hist = (-P_g*V_ig + Q_g*V_rg)/(V_rg**2 + V_ig**2)

        dI_igdV_rg = (Q_g*(V_ig**2 - V_rg**2) + 2*P_g*V_rg*V_ig)/(V_rg**2 + V_ig**2)**2
        dI_igdV_ig = (P_g*(V_ig**2 - V_rg**2) - 2*Q_g*V_rg*V_ig)/(V_rg**2 + V_ig**2)**2
        
        dI_igdQ_g  = V_rg/(V_rg**2 + V_ig**2)

        # fi_hist at J matrix
        fi_hist = - I_ig_hist + dI_igdV_rg*V_rg + dI_igdV_ig*V_ig + dI_igdQ_g*Q_g

        # Veq_hist at J matrix
        Veq_hist = V_rg**2 + V_ig**2 + (self.Vset)**2

        # Vrg Vig and Veq at J matrix
        self.stampJ(J_nonliniear, self.node_Vr, fr_hist)
        self.stampJ(J_nonliniear, self.node_Vi, fi_hist)
        self.stampJ(J_nonliniear, self.node_Q, Veq_hist)

        # 4 partial of I and V
        self.stampY(Y_nonlinear, self.node_Vr, self.node_Vr, dI_rgdV_rg)
        self.stampY(Y_nonlinear, self.node_Vr, self.node_Vi, dI_rgdV_ig)
        self.stampY(Y_nonlinear, self.node_Vi, self.node_Vr, dI_igdV_rg)
        self.stampY(Y_nonlinear, self.node_Vi, self.node_Vi, dI_igdV_ig)

        # 2 partial of I and Q
        self.stampY(Y_nonlinear, self.node_Vr, self.node_Q, dI_rgdQ_g)
        self.stampY(Y_nonlinear, self.node_Vi, self.node_Q, dI_igdQ_g)

        # 2 partial of Vrg and Vig
        self.stampY(Y_nonlinear, self.node_Q, self.node_Vr, 2*V_rg)
        self.stampY(Y_nonlinear, self.node_Q, self.node_Vi, 2*V_ig)

    def stamp_J(self, v, J_nonliniear):
        P_g   = self.P
        V_rg  = v[self.node_Vr]
        V_ig  = v[self.node_Vi]
        Q_g   = v[self.node_Q]

        # Real part
        I_rg_hist  = (-P_g*V_rg - Q_g*V_ig)/(V_rg**2 + V_ig**2)
        
        dI_rgdV_rg = (P_g*(V_rg**2 - V_ig**2) + 2*Q_g*V_rg*V_ig)/(V_rg**2 + V_ig**2)**2
        dI_rgdV_ig = (Q_g*(V_ig**2 - V_rg**2) + 2*P_g*V_rg*V_ig)/(V_rg**2 + V_ig**2)**2
        
        dI_rgdQ_g  = -V_ig/(V_rg**2 + V_ig**2)

        # fr_hist at J matrix
        fr_hist = - I_rg_hist + dI_rgdV_rg*V_rg + dI_rgdV_ig*V_ig + dI_rgdQ_g*Q_g

        # Imaginarry part
        I_ig_hist = (-P_g*V_ig + Q_g*V_rg)/(V_rg**2 + V_ig**2)

        dI_igdV_rg = (Q_g*(V_ig**2 - V_rg**2) + 2*P_g*V_rg*V_ig)/(V_rg**2 + V_ig**2)**2
        dI_igdV_ig = (P_g*(V_ig**2 - V_rg**2) - 2*Q_g*V_rg*V_ig)/(V_rg**2 + V_ig**2)**2
        
        dI_igdQ_g  = V_rg/(V_rg**2 + V_ig**2)

        # fi_hist at J matrix
        fi_hist = - I_ig_hist + dI_igdV_rg*V_rg + dI_igdV_ig*V_ig + dI_igdQ_g*Q_g

        # Veq_hist at J matrix
        Veq_hist = V_rg**2 + V_ig**2 + (self.Vset)**2

        # Vrg Vig and Veq at J matrix
        self.stampJ(J_nonliniear, self.node_Vr, fr_hist)
        self.stampJ(J_nonliniear, self.node_Vi, fi_hist)
        self.stampJ(J_nonliniear, self.node_Q, Veq_hist)

    def calc_Ir(self, 
                Vr: Union[PyomoVar, float],
                Vi: Union[PyomoVar, float],
                p:  Union[PyomoVar, float],
                q:  Union[PyomoVar, float],
                ):
        Ir_gen = (p* Vr + q * Vi) / (Vr ** 2 + Vi ** 2)
        return Ir_gen
    
    def calc_Ii(self, 
                Vr: Union[PyomoVar, float],
                Vi: Union[PyomoVar, float],
                p:  Union[PyomoVar, float],
                q:  Union[PyomoVar, float],
                ):
        Ii_gen = (p * Vi - q*Vr) / (Vr ** 2 + Vi ** 2)
        return Ii_gen






