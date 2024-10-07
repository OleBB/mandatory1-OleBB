import sympy as sp # type: ignore

# Define a symbolic variable
x = sp.symbols('x')

# Define a symbolic expression
expr = x**2 + 2*x + 1

# Simplify the expression
simplified_expr = sp.simplify(expr)

# Differentiate the expression with respect to x
derivative = sp.diff(expr, x)

# Integrate the expression with respect to x
integral = sp.integrate(expr, x)

# Solve the equation expr = 0
solutions = sp.solve(expr, x)

print("Original Expression:", expr)
print("Simplified Expression:", simplified_expr)
print("Derivative:", derivative)
print("Integral:", integral)
print("Solutions:", solutions)
