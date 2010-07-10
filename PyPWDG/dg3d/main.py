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
        from scipy.sparse import bmat, csr_matrix
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
        
        self.avd = mesh.average
        self.avn = mesh.jump/2
        self.jd = mesh.jump
        self.jn = mesh.average * 2
        self.b = mesh.boundary
        
        etof = mesh.etof
        nfs = len(numpy.hstack(etof))
        e1 = numpy.reshape(range(0,nfs), (len(etof), -1))
        etodf2 = numpy.hstack((e1, e1+nfs)) 
        self.etodf2 = etodf2
        edfi = numpy.hstack([numpy.ones(len(fs))*e for e,fs in enumerate(etof)])
        self.EDF2 = csr_matrix((numpy.ones(nfs*2), numpy.tile(edfi,2), range(0, 2*nfs+1)))
                
        
        self.bvd,self.bvn = mesh.vandermonde(self.quadpoints,self.fd,self.fn)
        np = self.bvd.get_shape()[0]
        adjustedweights =  mat(mesh.faceareas()).reshape(-1,1) * quadweights.reshape(1,-1)*2 #multiply by 2 because the quadrature is set up for a right-angle triangle, area 1/2
        self.awdiag = sparse.spdiags(adjustedweights.reshape(1,-1),0,np,np)
        weights = [quadweights * a*2 for a in mesh.faceareas()]
        blockvd, blockvn = mesh.localvandermondes(self.quadpoints, self.fd, self.fn)
        self.ip = innerproduct(blockvd + blockvn, weights*2)

    @print_timing
    def stiffness3(self, alpha = 1.0/2, beta = 1.0/2, delta = 1.0/2):
        """ Create a stiffness matrix for PWDG"""
        from scipy.sparse import bmat, csr_matrix
                
        print "Creating stiffness matrix - quickly?"
        jk = 1j * self.k
        jki = 1/jk
                
        internalf = bmat([[jk * alpha * self.jd, - self.avn ], [self.avd, - beta * jki * self.jn]])
        boundaryf = bmat([[jk*(1-delta)*self.b, - delta * self.b], [(1-delta)*self.b,- delta*jki*self.b]]) 
        
        b = createblock(internalf + boundaryf, self.ip.product)
        # now collapse the faces and contributions from D and N
        
        print b.get_shape()
        print self.EDF2.get_shape()

        EDF2es = expandsparse(self.EDF2, b.blocksize[0])        
        return EDF2es.transpose() * b * EDF2es
    
    @print_timing
    def stiffness2(self, alpha = 1.0/2, beta = 1.0/2, delta = 1.0/2):
        """ Create a stiffness matrix for PWDG"""
        from scipy.sparse import bmat, csr_matrix
                
        print "Creating stiffness matrix - quickly?"
        jk = 1j * self.k
        jki = 1/jk
                
        internalf = bmat([[jk * alpha * self.jd, - self.avn ], [self.avd, - beta * jki * self.jn]])
        boundaryf = bmat([[jk*(1-delta)*self.b, - delta * self.b], [(1-delta)*self.b,- delta*jki*self.b]]) 
        
        b = createblock(internalf + boundaryf, self.ip.product)
        # now collapse the faces and contributions from D and N
        
        print b.get_shape()
        print self.EDF2.get_shape()
        #return collapsebsr(self.etodf2, b)
        print "new prods"
        prod1 = lambda a,b : b
        prod2 = lambda a,b : a
        return sparseblockmultiply(self.EDF2.transpose(), sparseblockmultiply(b, self.EDF2, prod2), prod1)

    @print_timing
    def stiffness4(self, alpha = 1.0/2, beta = 1.0/2, delta = 1.0/2):
        """ Create a stiffness matrix for PWDG"""
        from scipy.sparse import bmat, csr_matrix
                
        print "Creating stiffness matrix - quickly?"
        jk = 1j * self.k
        jki = 1/jk
                
        internalf = bmat([[jk * alpha * self.jd, - self.avn ], [self.avd, - beta * jki * self.jn]])
        boundaryf = bmat([[jk*(1-delta)*self.b, - delta * self.b], [(1-delta)*self.b,- delta*jki*self.b]]) 
        
        b = createblock(internalf + boundaryf, self.ip.product)
        # now collapse the faces and contributions from D and N
        
        print b.get_shape()
        print self.EDF2.get_shape()
        #return collapsebsr(self.etodf2, b)
        print "new prods"
        prod1 = lambda a,b : b
        prod2 = lambda a,b : a
        return sparseblockmultiply2(self.EDF2.transpose(), sparseblockmultiply2(b, self.EDF2, prod2), prod1)
        
        
    @print_timing
    def stiffness(self, alpha = 1.0/2, beta = 1.0/2, delta = 1.0/2):
        from scipy.sparse import bmat
        """ Create a stiffness matrix for PWDG"""
                
        print "Creating stiffness matrix"
        jk = 1j * self.k
        jki = 1/jk
                        
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
              