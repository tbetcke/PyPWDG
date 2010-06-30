'''
Created on May 28, 2010

@author: joel
'''

from PyPWDG.dg3d.utils import *
import numpy

    

def solver(mesh, ndirs, nquad, k):
    """ Construct a solver 
    
    ndirs: number of directions to use on each element = 6 * ndirs^2
    nquad: quadrature degree for each face
    k: wave number
    """    
    return Solver(mesh, cubeRotations(cubeDirections(ndirs)), quadPoints(nquad), k)

class Solver(object):
    """ A Solver is capable of solving helmholtz equation """
    
    def __init__(self, mesh, directions, qpw, k):
        """ mesh - a mesh.Mesh object        
            directions - list of directions
            qpw - quadrature points and weights on unit triangle
            k - wave number
        """
            
        from scipy import sparse
        print "Initialising Assembler (vandermondes)"
        self.mesh = mesh
        self.directions = directions
        self.fd, self.fn = planeWaves(directions, k)
        self.nfns = len(directions)
        self.k = k
        self.quadpoints, quadweights = qpw
        nq = self.quadpoints.shape[0]
        self.average = expandsparse(mesh.average, nq)
        self.averagen = expandsparse(mesh.jump/2, nq)
        self.jump = expandsparse(mesh.jump, nq)
        self.jumpn = expandsparse(mesh.average * 2, nq)
        self.boundary = expandsparse(mesh.boundary, nq)
        self.bvd,self.bvn = mesh.vandermonde(self.quadpoints,self.fd,self.fn)
        np = self.bvd.get_shape()[0]
        adjustedweights =  mat(mesh.faceareas()).reshape(-1,1) * quadweights.reshape(1,-1)*2 #multiply by 2 because the quadrature is set up for a right-angle triangle, area 1/2
        self.awdiag = sparse.spdiags(adjustedweights.reshape(1,-1),0,np,np)
        
    @print_timing
    def stiffness(self, alpha = 1.0/2, beta = 1.0/2, delta = 1.0/2):
        """ Create a stiffness matrix for PWDG"""
                
        print "Creating stiffness matrix"
        jki = 1/(1j*self.k)
                    
        uhatie = self.average * self.bvd - beta * jki * self.jumpn * self.bvn
        sigmahatie = jki *self.averagen*self.bvn - alpha * self.jump * self.bvd
    #
        uhatbf = (1-delta)*self.boundary * self.bvd - delta*jki*self.boundary*self.bvn
        sigmahatbf = delta * jki * self.boundary * self.bvn - (1-delta)*self.boundary*self.bvd
        
        return self.bvn.H * self.awdiag * (uhatie + uhatbf) - 1j*self.k*self.bvd.H * self.awdiag * (sigmahatie + sigmahatbf)
   
    @print_timing
    def load(self, g, delta = 1.0/2):  
        """ Create a load matrix based on impedance boundary g"""
        
        print "Creating load matrix"
        import numpy.matlib  
        gp = numpy.matlib.vstack(self.mesh.values(self.quadpoints,g))
        jki = 1/(1j*self.k)
    
        return - delta * jki* self.bvn.H * self.awdiag * self.boundary * gp + (1-delta) * self.bvd.H * self.awdiag * self.boundary * gp  

    def solve(self, g):
        """ Solve the system based on impedance boundary g"""
        from scipy.sparse.linalg.dsolve.linsolve import spsolve 
        self.s = self.stiffness()
        self.l = self.load(g)
        print "Solving system"
        self.x = print_timing(spsolve)(self.s, self.l)
        self.g = g
        return self

    @print_timing
    def solutionvalues(self, refpoints):
        """ Values of the solution at refpoints on each element """
        bvd,bvn = self.mesh.vandermonde(refpoints, self.fd, self.fn)
        return (bvd * self.x, bvn * self.x)
              