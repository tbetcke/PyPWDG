'''
Created on Aug 26, 2010

@author: tbetcke

Provides objects for handling boundary data

'''

import numpy

class generic_boundary_data(object):
    """ Provides an interface for generic boundary data

        Initialize with
        bnd_data=generic_boundary_data(l_coeffs,r_coeffs,g,dg)
        
        Boundary data has the form:
        
        xi_1*u+xi_2*du/dn=mu_1*g+mu_2*dg/dn

        INPUT:
        l_coeffs - Input list [xi_1, xi_2]
        r_coeffs - Input list [mu_1, mu_2]

        g(xvals) is a function that takes an m x dim array and returns an array g(x)
        of length m.
        dg(xvals,n) takes additionallty a vector n of length dim, which is the normal
        direction at the face and returns an array of length m of normal derivatives
        
        dg does not necessarily have to be the normal derivative of g.
                
        if g=None and dg=None homogeneous boundary conditions are assumed
        
    """
    
    
    def __init__(self,l_coeffs,r_coeffs=None,g=None,dg=None):
        
        if g is None: g=lambda x: numpy.zeros(x.shape[0])
        if dg is None: dg= lambda x,n: numpy.zeros(x.shape[0])
        if r_coeffs is None: r_coeffs=[0, 0]
        self.__l_coeffs=l_coeffs
        self.__r_coeffs=r_coeffs
        self.__g=g
        self.__dg=dg
        
    def values(self,x,n=None):
        return self.__g(x)
    
    def derivs(self,x,n):
        return self.__dg(x,n)
            
    g=property(lambda self: self.__g)
    dg=property(lambda self: self.__dg)
    r_coeffs=property(lambda self: self.__r_coeffs)
    l_coeffs=property(lambda self: self.__l_coeffs)
    n=property(lambda self: 1)
    
    
    

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
        

        
        
    
    
        