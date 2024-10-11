
import numpy as np
import sympy as sp
import scipy.sparse as sparse
from matplotlib import cm
from matplotlib import pyplot as plt


x, y = sp.symbols('x,y')

class Poi:
        
    def __init__(self, L, ue):
        self.L = L
        self.ue = ue
        self.f = sp.diff(self.ue, x, 2)+sp.diff(self.ue, y, 2)
        
    def create_mesh(self, N):
        """Create 2D mesh and store in self.xij and self.yij"""
        self.N= N
        x = np.linspace(0, self.L, self.N+1) #oleb
        y = np.linspace(0, self.L, self.N+1) #oleb
        self.xij, self.yij = np.meshgrid(x, y, indexing='ij', sparse=True) #oleb
        return self.xij, self.yij #oleb
    
    def D2(self):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N+1, self.N+1), 'lil') #poisson1
        D[0, :4] = 2, -5, 4, -1 #poisson1
        D[-1, -4:] = -1, 4, -5, 2 #poissson1
        D /= self.dx**2 #poisson1
        return D #poisson1
    
    def __call__(self, N):
        kult = self.create_mesh(N)
        return kult
    
    def process_data(store_data, timestep, solution, h, l2):
        if store_data > 0:
            return {timestep: solution}
        elif store_data == -1:
            return (h, l2_error)
        else:
            raise ValueError("Invalid value for store_data; expected > 0 or == -1")

# Example usage:
store_data = 1
timestep = 10
solution = [1.2, 2.4, 3.6]  # Example solution
h = 0.1
l2_error = 0.005

result = Poi.process_data(store_data, timestep, solution, h, l2_error)
print(result)


""""
ue= x
#mesh=Poi(1,ue) 
L = 2.3
N = 10
mm = Poi(L,ue)(N)
print(mm)
"""