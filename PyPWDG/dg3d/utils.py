'''
Created on May 28, 2010

@author: joel
'''
import numpy
from numpy import  mat, identity, ones,hstack,vstack,zeros,dot
import scipy.special.orthogonal
import math


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

@print_timing
def createblock(mat, blocks):
    """ Creates a block sparse matrix
    
    mat gives the sparsity pattern.  The (i,j)th block is mat[i,j] * blocks(i,j)
    """
    from scipy.sparse import bsr_matrix as bsr_matrix
    csr = mat.tocsr().sorted_indices()
    csr.eliminate_zeros()
    zipip = zip(csr.indptr[:-1], csr.indptr[1:])
    coords = [(i,j) for i,p in enumerate(zipip) for j in csr.indices[p[0]:p[1]] ]
    data = numpy.array([mat.data[n] * blocks(i,j) for n, (i,j) in enumerate(coords)])
    s = csr.get_shape()
    b = data[0].shape
    return bsr_matrix((data, csr.indices,csr.indptr), shape=(s[0]*b[0],s[1]*b[1] ))

@print_timing
def sparseblockmultiply(a,b, prod=numpy.multiply):
    """ `multiply' a . b 
    
    a and b should have a compatible sparsity structure and at least one should be a block sparse
    matrix.  Effectively, we replace each entry x_ij of the non-block matrix, x by a_ij I where I
    is an appropriately sized identity matrix; and then perform the multiplication.
    
    you could make prod something like lambda a,b:a if you know that all the entries in the non-block are 1
    although it doesn't seem to make much difference
    
    the point is that we keep the block structure without having to explicitly construct a load of 
    dense identity matrices.  despite the fact that numpy does smart things when you multiply a matrix
    my the identity, this doesn't seem to carry through to the bsr world.
    
    As a side effect, if both a and b are block sparse then it performs a block by block sparse matrix multiply.  There
    is no reason to expect this to be more efficient than the inbuilt multiply.  However, we now have
    access to the prod function, which could therefore be distributed.  In any case, if a and b are both
    block sparse then prod should be set to something like numpy.dot, otherwise, the multiplication
    will be elementwise for each block multiply.  Note also that in this case, it's likely to be 
    easier just to use createblock
    
    as much as possible, the standard sparse matrix routines are used to do all the work on the underlying
    sparsity structure.  this might look a bit inefficient, but since all that's done in C++, it's not
    caveat refactorer - it's very easy to slow this down.
    """
    from scipy.sparse import csr_matrix, bsr_matrix, isspmatrix_bsr
    from scipy import int32
    # we want a and b to be either csr or bsr.  Also need to calculate the underlying sparsity
    # structure size
    ablocksize = (1,1)
    bblocksize = (1,1)
    if not isspmatrix_bsr(a): a = a.tocsr()
    else : ablocksize = a.blocksize
    if not isspmatrix_bsr(b): b = b.tocsr()
    else : bblocksize = b.blocksize
    ashape = [s/bs for s,bs in zip(a.get_shape(), ablocksize)]
    bshape = [s/bs for s,bs in zip(b.get_shape(), bblocksize)]
        
    a.sort_indices()
    # for each element of the product, we will iterate through a column of b.
    # it's much more efficient to do this if b is in column format.  
    # since b might be a block matrix and there's no blockcsr, we can't convert it
    # directly.  instead we build an index matrix:    
    bi = csr_matrix((range(0,len(b.data)), b.indices, b.indptr), dtype=int32, shape = bshape).tocsc()
    bi.sort_indices()
    
    # now determine the block-level sparsity structure of a . b
    ao = csr_matrix((numpy.ones(len(a.data)), a.indices, a.indptr), shape = ashape)
    bo = csr_matrix((numpy.ones(len(b.data)), b.indices, b.indptr), shape = bshape)
    abo = ao * bo
    abo.eliminate_zeros()
    abo.sum_duplicates()
    abo.sort_indices()
    
    # the data for the new matrix
    data = []
    indices = []
    indptr = [0]
    
    # iterate through each (i,j) element in abo
    for i, abop in enumerate(zip(abo.indptr[:-1], abo.indptr[1:])):
        for j in abo.indices[abop[0]:abop[1]]:
            # multiply the ith row of a by the jth column of b
            [pa,pa1] = a.indptr[i:i+2]
            [pb,pb1] = bi.indptr[j:j+2]
            ab = None
            while (pa < pa1 and pb < pb1):
                ak = a.indices[pa]
                bk = bi.indices[pb]
                if ak==bk: 
                    pab = prod(a.data[pa], b.data[bi.data[pb]])
                    if ab is None: ab = pab
                    else: ab = ab + pab
                if ak <= bk: pa+=1
                if bk <= ak: pb+=1
            data.append(ab) 
            indices.append(j)                
        indptr.append(len(indices))
    bs = data[0].shape    
    return bsr_matrix((numpy.array(data), indices, indptr), shape = (bs[0] * ashape[0], bs[1] * bshape[1]))    

class globalvandermonde(object):
    """ A class to calculate inner products and matrix vector based on local vandermonde matrices """
    def __init__(self, vandermondes, quadweights):
        """ vandermondes is a list of vandermonde matrices
            quadweights is a list of quadrature weights """  
        self.v = vandermondes
        self.w = quadweights
        # calculate the number of shape functions in each block
        self.nsf = [v.shape[1] for v in vandermondes]
    
    def product(self, i,j):
        """ Return the inner product of the ith face against the jth face """
    # It should be the case that w[i] = w[j], otherwise the
    # inner product makes no sense.
        return numpy.dot(numpy.multiply(self.v[i].H,self.w[i].A.flatten()), self.v[j])        

    def matvec(self, g):
        return lambda i,j : numpy.dot(self.v[i].H, numpy.multiply(self.w[i].reshape(-1,1), g[j]))

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


#  
#def boundaryintegral(mesh, quadorder, g):
#    [q,w] = quadPoints(quadorder)
#    gp = mat(boundaryvalues(mesh,q,g))
#    adjustedweights =  mat(faceareas(mesh)).reshape(-1,1) * w.reshape(1,-1)*2
##    print adjustedweights
##    print gp.reshape(adjustedweights.shape)
##    print [a * g.transpose() for (a,g) in zip(adjustedweights, gp.reshape(adjustedweights.shape))]
#    return adjustedweights.reshape(1,-1) * gp       