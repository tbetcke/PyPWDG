'''
Created on May 28, 2010

@author: joel
'''

from pypwdg.dg3d.utils import *
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
            
        from scipy.sparse import csr_matrix
        print "Initialising Assembler (vandermondes)"
        self.mesh = mesh
        self.directions = directions
        self.fd, self.fn = planeWaves(directions, k)
        self.nfns = len(directions)
        self.k = k
        self.quadpoints, quadweights = qpw
        
        # averages, jumps and boundary
        self.avd = mesh.average
        self.avn = mesh.jump/2
        self.jd = mesh.jump
        self.jn = mesh.average * 2
        self.b = mesh.boundary
        
        # EDF2 - element to 2 2-sided face matrices.  Used to collapse the vandermondes for each face for 
        # each element for both function vals and derivatives  
        nfs = len(numpy.hstack(mesh.etof))
        edfi = numpy.hstack([numpy.ones(len(fs))*e for e,fs in enumerate(mesh.etof)])
        self.EDF2 = csr_matrix((numpy.ones(nfs*2), numpy.tile(edfi,2), range(0, 2*nfs+1)))
                        
        weights = [quadweights * a*2 for a in mesh.faceareas()]*2
        blockvd, blockvn = mesh.localvandermondes(self.quadpoints, self.fd, self.fn)
        self.vandermondes = globalvandermonde(blockvd + blockvn, weights)

    @print_timing
    def stiffness(self, alpha = 1.0/2, beta = 1.0/2, delta = 1.0/2):
        """ Create a stiffness matrix for PWDG"""
        from scipy.sparse import bmat
        from pypwdg.utils.sparse import createvbsr
                
        print "Creating stiffness matrix - quickly?"
        jk = 1j * self.k
        jki = 1/jk
                
        internalf = bmat([[jk * alpha * self.jd, - self.avn             ], 
                          [self.avd,             - beta * jki * self.jn ]])
        boundaryf = bmat([[jk*(1-delta)*self.b, - delta * self.b    ], 
                          [(1-delta)*self.b,    - delta*jki*self.b  ]]) 
        
#        S = createblock(internalf + boundaryf, self.vandermondes.product)
        S = createvbsr(internalf + boundaryf, self.vandermondes.product, self.vandermondes.nsf, self.vandermondes.nsf)
        
        # now collapse the faces and contributions from D and N
#        return sparseblockmultiply(self.EDF2.transpose(), sparseblockmultiply(S, self.EDF2))
        return (S * self.EDF2).__rmul__(self.EDF2.transpose())
  
    @print_timing
    def load(self, g, delta = 1.0/2):  
        """ Create a load matrix based on impedance boundary g"""
        
        print "Creating load matrix"
        from scipy.sparse import bmat, csr_matrix
        from pypwdg.utils.sparse import createvbsr

        gp = self.mesh.values(self.quadpoints,g)
        jki = 1/(1j*self.k)
        
        impedance = bmat([[(1-delta) * self.b   ],
                          [-delta * jki * self.b]])
    
        # this is long-winded.  The point is that it allows for the vandermondes to be managed, which,
        # in principle means that they could be distributed.  If one didn't care then the best way would
        # be to build a block sparse matrix out of the vandermondes and a column vector from gp
#        L = createblock(impedance, self.vandermondes.matvec(gp))
        L = createvbsr(impedance, self.vandermondes.matvec(gp), self.vandermondes.nsf, numpy.ones(len(gp)))
        
        
        # the final ones matrix pulls all the gs from each element into one place
#        return sparseblockmultiply(sparseblockmultiply(self.EDF2.transpose(), L), csr_matrix(numpy.ones((len(gp),1))))
        return L.__rmul__(self.EDF2.transpose()) * csr_matrix(numpy.ones((len(gp),1)))
    
#        return - delta * jki* self.bvn.H * self.awdiag * self.boundary * gp + (1-delta) * self.bvd.H * self.awdiag * self.boundary * gp  

    def solve(self, g):
        """ Solve the system based on impedance boundary g"""
        from scipy.sparse.linalg.dsolve.linsolve import spsolve 
        self.s = self.stiffness()
        self.l = self.load(g)
        print "Solving system"
        self.x = print_timing(spsolve)(self.s.tocsr(), self.l.todense())
        self.g = g
        return self

    @print_timing
    def solutionvalues(self, refpoints):
        """ Values of the solution at refpoints on each element """
        bvd,bvn = self.mesh.vandermonde(refpoints, self.fd, self.fn)
        return (bvd * self.x, bvn * self.x)
              