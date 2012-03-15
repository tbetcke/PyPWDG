'''
Created on Aug 26, 2010

@author: tbetcke

Provides objects for handling boundary data

'''

import numpy
import collections as c

class BoundaryCondition(object):
    def __init__(self, lcoeffs, rcoeffs = None, data = None):
        self.data = ZeroBoundaryData() if data is None else data
        self.coeffs = BoundaryCoefficients(lcoeffs, rcoeffs)

class ZeroBoundaryData(object):
    def __init__(self, g):
        self.g = g
        
    def values(self,x):
        return numpy.zeros((x.shape[0],1)) 
    
    def derivs(self,x,n):
        return numpy.zeros((x.shape[0],1)) 

class BoundaryCoefficients(object):
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
    
    
    def __init__(self,l_coeffs,r_coeffs=None):  
        if r_coeffs is None: r_coeffs=[0, 0]
        self.l_coeffs=l_coeffs
        self.r_coeffs=r_coeffs
            
#    g=property(lambda self: lambda x: numpy.zeros(x.shape[0]) if self.__g is None else self.__g)
#    dg=property(lambda self: lambda x,n: numpy.zeros(x.shape[0]) if self.__dg is None else self.__dg)
#    r_coeffs=property(lambda self: self.__r_coeffs)
#    l_coeffs=property(lambda self: self.__l_coeffs)
#    n=property(lambda self: 1)
def generic_boundary_data(lc, rc, g):
    return BoundaryCondition(lc, rc, g)

def zero_impedance(k):
    """ Zero impedance boundary conditions du/dn-iku=0    
    
        bnd_data=zero_impedance(k), where k is the wavenumber
        
    """
    return BoundaryCondition([-1j*k, 1])
        
def zero_dirichlet():
    """ Zero Dirichlet boundary conditions u=0 

        bnd_data=zero_dirichlet()
        
    """
    
    return BoundaryCondition([1, 0])
        
def zero_neumann():
    """ Zero Neumann boundary conditions u=0 

        bnd_data=zero_dirichlet()
        
    """
    return BoundaryCondition([0,1])
        
def dirichlet(g):
    return BoundaryCondition([1,0],[1,0],g)
        
def neumann(g):
    return BoundaryCondition([0,1],[0,1],g)
        
        
    
    
        