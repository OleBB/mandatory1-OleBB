import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm

x, y, t = sp.symbols('x,y,t')

class Wave2D:

    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        self.N= N #opg1
        self.h = self.L/self.N #opg1
        x = np.linspace(0, self.L, self.N+1) #opg1
        y = np.linspace(0, self.L, self.N+1) #opg1
        self.xji, self.yij = np.meshgrid(x, y, indexing='ij', sparse=sparse) #lecture 7  

    def D2(self, N):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N+1, self.N+1), 'lil') #wave1D.py
        D[0, :4] = 2, -5, 4, -1 #lecture 7
        D[-1, -4:] = -1, 4, -5, 2 #lecture 7
        D /= self.h**2 #poisson2d.py
        return D #wave1D.py

    @property
    def w(self):
        """Return the dispersion coefficient"""
        kx = self.mx*self.pi #oppgaveteksten
        ky = self.my*self.pi #oppgaveteksten sier ky = my*pi
        return self.c*(kx+ky) #what about plus minus?

    def ue(self, mx, my):
        """Return the exact standing wave"""
        return sp.sin(mx*sp.pi*x)*sp.sin(my*sp.pi*y)*sp.cos(self.w*t)

    def initialize(self, N, mx, my):
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        Unp1[0] = 0

    @property
    def dt(self):
        """Return the time step"""
        return self.cfl*self.h/self.c #wave1d.py #oleb: dx til h

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        raise NotImplementedError

    def apply_bcs(self):
        raise NotImplementedError

    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.N = N #denne finnes i meshgrid
        self.cfl = cfl
        self.Nt = Nt
        self.c = c 
        self.mx = mx
        self.my = my
        self.store_data = store_data
        if store_data > 0:
            return {timestep: solution} #gpt
        elif store_data == -1:
            return (self.h, self.l2_error) #gpt
        else:
            return ValueError("Invalid value for store_data")

    def convergence_rates(self, m=4, cfl=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

class Wave2D_Neumann(Wave2D):

    def D2(self, N):
        raise NotImplementedError

    def ue(self, mx, my):
        raise NotImplementedError

    def apply_bcs(self):
        raise NotImplementedError

def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 1e-2

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 0.05

def test_exact_wave2d():
    raise NotImplementedError
