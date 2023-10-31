from pyomo.environ import *

# Define network data
num_buses = 4
bus_data = {
    1: {'v_min': 0.95, 'v_max': 1.05},
    2: {'v_min': 0.95, 'v_max': 1.05},
    3: {'v_min': 0.95, 'v_max': 1.05},
    4: {'v_min': 0.95, 'v_max': 1.05},
}
line_data = {
    (1, 2): {'r': 0.01, 'x': 0.1, 'b': 0.05},
    (1, 3): {'r': 0.02, 'x': 0.2, 'b': 0.1},
    (2, 3): {'r': 0.015, 'x': 0.15, 'b': 0.075},
    (3, 4): {'r': 0.01, 'x': 0.1, 'b': 0.05},
}

# Define measurement data
v_r_meas = {1: 1.0, 2: 1.02, 3: 0.98, 4: 1.0}
v_i_meas = {1: 0.0, 2: 0.2, 3: -0.1, 4: 0.0}

# Define the Pyomo model
model = ConcreteModel()

# Define the decision variables

model.v_r = Var(range(1, num_buses+1), bounds=(bus_data[i]['v_min'], bus_data[i]['v_max']))
model.v_i = Var(range(1, num_buses+1), bounds=(bus_data[i]['v_min'], bus_data[i]['v_max']))
model.n_r = Var(range(1, num_buses+1))
model.n_i = Var(range(1, num_buses+1))

# Define the objective function to minimize noise
model.noise = Objective(expr=sum(model.n_r[i]**2 + model.n_i[i]**2 for i in range(1, num_buses+1)))

# Define the network equations
for i in range(1, num_buses+1):
    # Real power flow equation
    model.power_balance_r = Constraint(expr=sum(model.v_r[j] * (cos(model.v_ang[i] - model.v_ang[j])*line_data[(i,j)]['r'] + sin(model.v_ang[i] - model.v_ang[j])*line_data[(i,j)]['x']) for j in range(1, num_buses+1) if (i,j) in line_data.keys()) + model.n_r[i] == 0)
    # Imaginary power flow equation
    model.power_balance_i = Constraint(expr=sum(model.v_i[j] * (cos(model.v_ang[i] - model.v_ang[j])*line_data[(i,j)]['x'] - sin(model.v_ang[i] - model.v_ang[j])*line_data[(i,j)]['r']) for j in range(1, num_buses+1) if (i,j) in line_data.keys()) + model.n_i[i] == 0)
    # Voltage magnitude constraint
    model.voltage_mag = Constraint(expr=model.v_r[i]**2 + model.v_i[i]**2 == model.v_mag[i]**2)

# Solve the Pyomo model using the GS4 solver
results = gs4_solve(model)

# Extract the solution to the state estimation
