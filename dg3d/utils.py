'''
Created on May 28, 2010

@author: joel
'''
import numpy
from numpy import  mat, identity, ones,hstack,vstack,zeros,dot,asmatrix
import scipy.special.orthogonal
import math
    
def normalM(points):
    """ Compute a normal to the plane intersecting the points in the rows of M"""
    dim = points.shape[1] # dimension - might as well keep this general
    # M consists of the differences between rows i and 0 for i = 1..dim plus a row of zeros
    # The row of zeros is necessary to make the QR factorisation produce the "missing" normal;
    M = dot(vstack( (hstack( (-ones((dim-1,1)),identity(dim-1, float)) ),zeros((1,dim))) ),points) 
    q = numpy.linalg.qr(M.transpose())[0]
    return q[:,dim-1].transpose()

def expandsparse(csr, n):
    """ Returns an exanded sparse matrix. Each entry, a, in csr is replaced by a*I(n,n) """
    from scipy import sparse
    zipip = zip(csr.indptr[:-1], csr.indptr[1:])

    #repeat each row of data n times 
    ndata = hstack([numpy.tile(csr.data[ip[0]:ip[1]],n) for ip in zipip])
    #repeat and shift each row of indices n times
    nindices = [n*i + k for ip in zipip for k in range(n) for i in csr.indices[ip[0]:ip[1]]  ]
    #repeat the jumps in the row pointers n times
    ipstep = [ip[1] - ip[0] for ip in zipip]
    nindptr=[0]
    for step in numpy.array(ipstep).repeat(n): nindptr.append(nindptr[-1]+step)
    return sparse.csr_matrix((ndata,nindices,nindptr),shape = (csr.get_shape()[0]*n,csr.get_shape()[1]*n))

def cubeDirections(n):
    """ Return n^2 directions roughly parallel to (1,0,0)"""
    
    r = [2.0*t/(n+1)-1 for t in range(1,n+1)]
    return [v / math.sqrt(dot(v,v)) for v in [numpy.array([1,y,z]) for y in r for z in r]]

def cubeRotations(directions):
    """ Rotate each direction through the faces of the cube"""
    M = mat(directions)
    return numpy.vstack([numpy.vstack([M,-M])[:,i] for i in [(0,1,2),(1,2,0),(2,0,1)] ])

def planeWaves(directions, k):
    """ Generate plane wave shape functions"""
    fd = lambda x, n: numpy.exp(1j * k * x * directions.transpose())
    fn = lambda x, n: 1j*k*numpy.multiply(n * directions.transpose(), fd(x,n)) 
    return (fd,fn)
    
def quadPoints(n):
    """ Degree n quadrature points on a triangle"""
    x00,w00 = scipy.special.orthogonal.p_roots(n)
    x01,w01 = scipy.special.orthogonal.j_roots(n,1,0)
    x00s = mat(x00+1)/2
    x01s = mat(x01+1)/2
    w = mat(w01).transpose() * mat(w00) / 8 # a factor of 2 for the legendres and 4 for the jacobi10
    x = x01s.transpose() * ones(x00s.shape)
    y = (1-x01s.transpose()) * x00s
    return hstack((x.reshape(-1,1), y.reshape(-1,1))), w.reshape(-1,1)
  
def impedance(dir,k):
    """ Returns a function giving impendance boundary conditions.
    
    The boundary conditions are for a plane wave of direction dir and wave number k """
    fd,fn = planeWaves(numpy.matrix(dir),k)
    return lambda x,n: fn(x,n) + 1j * k *fd(x,n)


def print_timing(func):
    """ timing utility function.  Use @print_timing """
    import time
    def wrapper(*arg):
        t1 = time.time()
        res = func(*arg)
        t2 = time.time()
        print '%s took %0.3f ms' % (func.func_name, (t2-t1)*1000.0)
        return res
    return wrapper
#  
#def boundaryintegral(mesh, quadorder, g):
#    [q,w] = quadPoints(quadorder)
#    gp = mat(boundaryvalues(mesh,q,g))
#    adjustedweights =  mat(faceareas(mesh)).reshape(-1,1) * w.reshape(1,-1)*2
##    print adjustedweights
##    print gp.reshape(adjustedweights.shape)
##    print [a * g.transpose() for (a,g) in zip(adjustedweights, gp.reshape(adjustedweights.shape))]
#    return adjustedweights.reshape(1,-1) * gp       