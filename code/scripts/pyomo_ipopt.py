from pyomo.environ import *

class StateEstimation:
    def create_model(self,
                     model: ConcreteModel,
                     grid_data: dict) -> ConcreteModel:
        num_busses = 0
        for ele in bus:
            if ele.Bus >= num_busses:
                 num_busses = ele.Bus

    def create_parameters(self,
                     model: ConcreteModel,
                     grid_data: dict) -> ConcreteModel:
         pass
              
    
    def create_constraints(self,
                     model: ConcreteModel,
                     grid_data: dict) -> ConcreteModel:
         pass
            
    def create_objective(self,
                     model: ConcreteModel,
                     grid_data: dict) -> ConcreteModel:
         
         def minimize_noise(model):
              """
              This function should return objective to minimize the noise
              at all RTU

              """
              


