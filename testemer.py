
import numpy as np
import sympy as sp
import scipy.sparse as sparse
from matplotlib import cm
from matplotlib import pyplot as plt


def mesh2D(Nx, Ny, Lx, Ly, sparse=False):
    x = np.linspace(0, Lx, Nx+1)
    y = np.linspace(0, Ly, Ny+1)
    return np.meshgrid(x, y, indexing='ij', sparse=sparse)

xij, yij = mesh2D(N, N, 1, 1, False)
u2 = xij*(1-xij)*yij*(1-yij)
plt.figure(figsize=(3, 2))
plt.contourf(xij, yij, u2);

def D2(self,N):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil') #poisson1
        D[0, :4] = 2, -5, 4, -1 #poisson1
        D[-1, -4:] = -1, 4, -5, 2 #poissson1
        D /= self.dx**2 #poisson1
        return D #poisson1

#STJÅLEN ifrå lecture 7
def solver(N, L, Nt, cfl=0.5, c=1, store_data=10, u0=lambda x, y: np.exp(-40*((x-0.6)**2+(y-0.5)**2))):
    xij, yij = mesh2D(N, N, L, L)
    Unp1, Un, Unm1 = np.zeros((3, N+1, N+1))
    Unm1[:] = u0(xij, yij)
    dx = L / N
    D = D2(N)/dx**2
    dt = cfl*dx/c
    Un[:] = Unm1[:] + 0.5*(c*dt)**2*(D @ Unm1 + Unm1 @ D.T)
    plotdata = {0: Unm1.copy()}
    if store_data == 1:
        plotdata[1] = Un.copy()
    for n in range(1, Nt):
        Unp1[:] = 2*Un - Unm1 + (c*dt)**2*(D @ Un + Un @ D.T)
        # Set boundary conditions
        Unp1[0] = 0
        Unp1[-1] = 0
        Unp1[:, -1] = 0
        Unp1[:, 0] = 0
        # Swap solutions
        Unm1[:] = Un
        Un[:] = Unp1
        if n % store_data == 0:
            plotdata[n] = Unm1.copy() # Unm1 is now swapped to Un
    return xij, yij, plotdata

xij, yij, data = solver(40, 1, 171, cfl=0.71, store_data=5)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(xij, yij, data[0], cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)