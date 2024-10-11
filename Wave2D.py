import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
from matplotlib import rc

x, y, t = sp.symbols('x,y,t')

class Wave2D:

    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        self.N = N
        self.L = 1
        self.h = self.L/self.N #opg1
        x = np.linspace(0, self.L, self.N+1) #opg1
        y = np.linspace(0, self.L, self.N+1) #opg1
        self.xij, self.yij = np.meshgrid(x, y, indexing='ij', sparse=sparse) #lecture 7  

    def D2(self, N):
        """Return second order differentiation matrix"""
        self.N = N
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N+1, self.N+1), 'lil') #wave1D.py (men N+1??)
        D[0, :4] = 2, -5, 4, -1 #lecture 7
        D[-1, -4:] = -1, 4, -5, 2 #lecture 7
        D /= self.h**2 #poisson2d.py ( / ?)
        return D #wave1D.py

    @property
    def w(self):
        """Return the dispersion coefficient"""
        kx = self.mx*np.pi #oppgaveteksten
        ky = self.my*np.pi #oppgaveteksten sier ky = my*pi
        k_abs = np.sqrt(kx**2 +ky**2) # 
        return self.c*(k_abs) 

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
        self.N = N
        self.mx = mx
        self.my = my
        self.Unm1, self.Un, self.Unp1 = np.zeros((3, self.N+1, self.N+1)) #oleb endret  . N+ eller ikke pluss?
        self.Unm1[:] = sp.lambdify((x, y, t), self.ue(mx, my))(self.xij, self.yij,0)

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
        dx = self.L/(self.N)
        dy = self.L/(self.N)
        UE = sp.lambdify((x,y,t), self.ue(self.mx,self.my))(self.xij, self.yij,t0)
        l2_error = np.sqrt(dx*dy*np.sum((UE - u)**2))
        return l2_error

    def apply_bcs(self):
        self.Unp1[0] = 0
        self.Unp1[-1] = 0
        self.Unp1[:, -1] = 0
        self.Unp1[:, 0] = 0

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
        
        self.cfl = cfl
        self.Nt = Nt
        self.c = c 
        self.mx = mx
        self.my = my
        self.store_data = store_data

        self.create_mesh(N); # 
        self.initialize(N, mx, my) #siomn
        D = self.D2(N)#/self.h**2 # 
        self.Un[:] = self.Unm1[:] + .5*(self.c*self.dt)**2*(D @ self.Unm1 + self.Unm1 @ D.T)
        
        plotdata = {0: self.Unm1.copy()};  # 
        l2_error = [] # 
        t = self.dt # 

        if store_data == 1:
            plotdata[1] = self.Un.copy()
            l2_error.append(self.l2_error(self.Un, t))
        for n in range(1, Nt):
            self.Unp1[:] = 2*self.Un - self.Unm1 + (c*self.dt)**2*(D @ self.Un + self.Un @ D.T)
            
            self.apply_bcs()
            self.Unm1[:] = self.Un
            self.Un[:] = self.Unp1
            t += self.dt
            if n % store_data == 0: 
                plotdata[n] = self.Unm1.copy() #Unm1 byttet 
                l2_error.append(self.l2_error(self.Un, t))
        if store_data == -1: 
            return self.h, l2_error 
        else:
            return plotdata

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
            print(f"m: {m,}")
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

class Wave2D_Neumann(Wave2D):

    def D2(self, N):
        self.N = N
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N+1, self.N+1), 'lil')
        D[0, :2] = -2, 2
        D[-1, -2:] = 2, -2
        D = D/self.h**2
        return D
        

    def ue(self, mx, my):
        ue = sp.cos(mx*sp.pi*x)*sp.cos(my*sp.pi*y)*sp.cos(self.w*t)
        return ue
        
    def apply_bcs(self):
        pass
        
    def Animate(self, N, Nt = 10, mx = 2, my = 2):
        data = self(N, Nt, mx = mx, my = my, store_data = 1)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        frames = []
        for n, val in data.items():
            
            frame = ax.plot_surface(self.xij, self.yij, val, vmin=-0.5*data[0].max(),
                                    vmax=data[0].max(), cmap=cm.YlGn,
                                    linewidth=0, antialiased=False)
            frames.append([frame])

        animate = animation.ArtistAnimation(fig, frames, interval=400, blit=True, repeat_delay=1000)
        animate.save('NeumannWave.gif', writer='pillow', fps=12)

def test_convergence_wave2d():
    solD = Wave2D()
    r, E, h = solD.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 1e-2

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 0.05

def test_exact_wave2d():
    solD = Wave2D()
    r, E, h = solD.convergence_rates(m = 4, cfl = 1/np.sqrt(2), mx = 1, my = 1)
    assert abs(E[-1]) < 1e-6

def seeAnimation():
    Wave2D_Neumann().Animate(32, 48)
seeAnimation()