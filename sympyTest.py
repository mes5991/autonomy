import sympy

x,y,theta,theta_dot,v,acc,L,phi,dt = sympy.symbols("x,y,theta,theta_dot,v,acc,L,phi,dt")

f = sympy.Matrix([[x + (v * sympy.cos(theta) * dt) + (.5 * acc * sympy.cos(theta) * dt ** 2)],
                  [y + (v * sympy.sin(theta) * dt) + (.5 * acc * sympy.sin(theta) * dt ** 2)],
                  [theta + (v * sympy.tan(phi) * dt)/L],
                  [acc * dt],
                  [(v * sympy.tan(phi) * dt)/L]])

stateVector = sympy.Matrix([x,y,theta,v,theta_dot])

j = f.jacobian(stateVector)

sympy.pprint(f)
sympy.pprint(stateVector)
sympy.pprint(j)
