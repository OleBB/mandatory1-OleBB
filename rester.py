#rester fra kode

#rester fra create_mesh i poisson2d.py
        xi = self.px.create_mesh(self.N+1) #oleb
        yj = self.py.create_mesh(self.N+1) #oleb
        self.xij, self.yij = np.meshgrid(xi, yj, indexing='ij', sparse=True) #oleb
        return self.xij, self.yij #oleb


