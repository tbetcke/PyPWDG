'''
Created on Aug 26, 2010

@author: tbetcke

Provides objects for handling boundary data

'''

import numpy

class generic_boundary_data(object):
    """ Provides an interface for generic boundary data

        Initialize with
        bnd_data=generic_boundary_data(l_coeffs,r_coeffs,g)
        
        Boundary data has the form:
        
        xi_1*u+xi_2*du/dn=mu_1*g+mu_2*dg/dn

        INPUT:
        l_coeffs - Input list [xi_1, xi_2]
        r_coeffs - Input list [mu_1, mu_2]

        g should support the basis interface for one shape function
                
        if g=None homogeneous boundary conditions are assumed
        
    """
    
    
    def __init__(self,l_coeffs,r_coeffs=None,g = None):        
        if r_coeffs is None: r_coeffs=[0, 0]
        self.l_coeffs=l_coeffs
        self.r_coeffs=r_coeffs
        self.g = g
        self.n = 1
        
    def values(self,x,n=None):
        return numpy.zeros((x.shape[0],1)) if self.g is None else self.g.values(x,n)
    
    def derivs(self,x,n):
        return numpy.zeros((x.shape[0],1)) if self.g is None else self.g.derivs(x,n)
            
#    g=property(lambda self: lambda x: numpy.zeros(x.shape[0]) if self.__g is None else self.__g)
#    dg=property(lambda self: lambda x,n: numpy.zeros(x.shape[0]) if self.__dg is None else self.__dg)
#    r_coeffs=property(lambda self: self.__r_coeffs)
#    l_coeffs=property(lambda self: self.__l_coeffs)
#    n=property(lambda self: 1)
    
    
    

class zero_impedance(generic_boundary_data):
    """ Zero impedance boundary conditions du/dn-iku=0    
    
        bnd_data=zero_impedance(k), where k is the wavenumber
        
    """
    
    def __init__(self,k):
        super(zero_impedance,self).__init__([-1j*k, 1])
        
class zero_dirichlet(generic_boundary_data):
    """ Zero Dirichlet boundary conditions u=0 

        bnd_data=zero_dirichlet()
        
    """
    
    def __init__(self):
        super(zero_dirichlet,self).__init__([1, 0])
        
class zero_neumann(generic_boundary_data):
    """ Zero Neumann boundary conditions u=0 

        bnd_data=zero_dirichlet()
        
    """
    
    def __init__(self):
        super(zero_neumann,self).__init__([0,1])
        
class dirichlet(generic_boundary_data):
    """ Dirichlet boundary conditions u=g

        bnd_data=dirichlet(g)
        
    """
    def __init__(self,g):
        super(dirichlet,self).__init__([1,0],[1,0],g)
        
class neumann(generic_boundary_data):
    """ Neumann boundary conditions u_n=g

        bnd_data=neumann(g)
        
    """
    def __init__(self,g):
        super(neumann,self).__init__([0,1],[0,1],g)
        
        
    
    
        