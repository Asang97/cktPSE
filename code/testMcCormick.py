from pyomo.environ import *
from testAddMcCormickEnvelope import add_McCormick 
"Testing McCormick with wiki example"

if True:
    m = ConcreteModel()
    m.x = Var()
    m.y = Var()
    
    "set upper and lower bound"
    x_l = 0
    x_u = 1.5
    y_l = 0
    y_u = 1.5
    
    "set objective"

    m.obj = Objective(expr= -m.x + m.x*m.y - m.y, sense = minimize)
    
    "initialize constraint list"
    m.consl = ConstraintList()
    # m.consl.add(expr= m.x*m.y +m.x + m.y == 8)
    m.consl.add(expr= -6*m.x + 8*m.y - 3 <= 0)
    m.consl.add(expr= 3*m.x - m.y - 3 <= 0)

    m.x = -45
    m.y = -44
    "Constrct McCormick Envelope"
    "Though it is convex envelope, note if" 
    "initial from outside the bound,"
    "we lost the convexity"
    
    if True:
        add_McCormick(x_l, x_u, y_l, y_u,m)

    if False:
        m.w = Var()
        m.consl.add(expr= m.w +m.x + m.y == 8)
        m.consl.add(expr= m.x*m.y == m.w)
        m.consl.add(expr= x_l*m.y + m.x*y_l - x_l*y_l <= m.w)
        m.consl.add(expr= x_u*m.y + m.x*y_u - x_u*y_u <= m.w)
        m.consl.add(expr= x_u*m.y + m.x*y_l - x_u*y_l >= m.w)
        m.consl.add(expr= m.x*y_u + x_l*m.y - x_l*y_u >= m.w)
        m.consl.add(expr= m.x<=x_u)
        m.consl.add(expr= m.x>=x_l)
        m.consl.add(expr= m.y<=y_u)
        m.consl.add(expr= m.y>=y_l)

    solver = SolverFactory('ipopt')
    result = solver.solve(m)
    m.solutions.store_to(result)
    result.write()
