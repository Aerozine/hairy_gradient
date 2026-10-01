import sympy as sp
x,y= sp.symbols('x y')
f1 = 5*(x + 1)**2 - 4*x*y + 3*y**2 - 2*x + 7*y + 2
f2 = (0.2*x**2 + 0.3*y**2 - 0.2*x*y - 15*sp.cos(0.7*x)
      - 9*sp.sin(0.6*y - 0.7*x) + 3*x)

for i, f in enumerate([f1, f2], start=1):
    grad = sp.Matrix([f.diff(x), f.diff(y)])
    hess = sp.hessian(f, (x, y))
    print(40*'-')
    sp.pprint(grad)
    sp.pprint(hess)
    print("LATEX : ")
    print(f'\\nabla f_{i} = {sp.latex(grad)}')
    print(f'\\nabla^2 f_{i} = {sp.latex(hess)}')
