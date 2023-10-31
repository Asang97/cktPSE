from pyomo.environ import *


def add_McCormick(x_l, x_u, y_l, y_u, m):
    m.w = Var()
    # m.consl.add(expr= m.w -m.x - m.y == 0)
    # m.consl.add(expr= m.x*m.y == m.w)
    m.consl.add(expr= x_l*m.y + m.x*y_l - x_l*y_l <= m.w)
    m.consl.add(expr= x_u*m.y + m.x*y_u - x_u*y_u <= m.w)
    m.consl.add(expr= x_u*m.y + m.x*y_l - x_u*y_l >= m.w)
    m.consl.add(expr= m.x*y_u + x_l*m.y - x_l*y_u >= m.w)
    m.consl.add(expr= m.x<=x_u)
    m.consl.add(expr= m.x>=x_l)
    m.consl.add(expr= m.y<=y_u)
    m.consl.add(expr= m.y>=y_l)
    pass