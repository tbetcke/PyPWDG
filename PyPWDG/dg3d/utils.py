'''
Created on May 28, 2010

@author: joel
'''
import numpy
from numpy import  mat, identity, ones,hstack,vstack,zeros,dot,asmatrix
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
    return bsr_matrix((data, csr.indices,csr.indptr))

@print_timing
def sparseblockmultiply2(a,b, prod=numpy.multiply):
    from scipy.sparse import csr_matrix, bsr_matrix, isspmatrix_bsr, coo_matrix
    from scipy import int32
    # we want a and b to be either csr or bsr
    if not isspmatrix_bsr(a): a = a.tocsr()
    if not isspmatrix_bsr(b): b = b.tocsr()
    a.sort_indices()
    # for each element of the product, we will iterate through a column of b.
    # it's much more efficient to do this if b is in column format.  
    # since b might be a block matrix and there's no blockcsr, we can't convert it
    # directly.  instead we build an index matrix:    
    bi = csr_matrix((range(0,len(b.data)), b.indices, b.indptr), dtype=int32).tocsc()
    bi.sort_indices()
    
    # now determine the block-level sparsity structure of a . b
    ao = csr_matrix((numpy.ones(len(a.data)), a.indices, a.indptr))
    bo = csr_matrix((numpy.ones(len(b.data)), b.indices, b.indptr))
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
        
    return bsr_matrix((numpy.array(data), indices, indptr))    

                        
    

@print_timing
def sparseblockmultiply(a,b, prod=numpy.multiply):
    """ `multiply' a . b 
    
    a and b should have a compatible sparsity structure and at least one should be a block sparse
    matrix.  Effectively, each entry a_ij of the non-block matrix is replaced with a_ij I where I
    is an appropriately sized identity matrix.
    
    As a side effect, if both a and b are block sparse then it performs a block by block sparse matrix multiply.  There
    is no reason to expect this to be more efficient than the inbuilt multiply.  However, we now have
    access to the prod function, which could therefore be distributed.  In any case, if a and b are both
    block sparse then prod should be set to something like numpy.dot, otherwise, the multiplication
    will be elementwise for each block multiply.  Note also that in this case, it's likely to be 
    easier just to use createblock
    """
    import time 
    from scipy.sparse import csr_matrix, bsr_matrix, isspmatrix_bsr, coo_matrix
    from scipy import int32
    t0 = time.time()
    print "t1",time.time()-t0
    # we want a and b to be either csr or bsr
    if not isspmatrix_bsr(a): a = a.tocsr()
    if not isspmatrix_bsr(b): b = b.tocsr()
    print "t2",time.time()-t0    
    # ao and bo just contain the sparsity structure of a and b
    ao = csr_matrix((numpy.ones(len(a.data)), a.indices, a.indptr))
    bo = csr_matrix((numpy.ones(len(b.data)), b.indices, b.indptr))
    print "t3",time.time()-t0    
    # abo gives us the sparsity structure of a . b.
    abo = ao*bo
    abo.eliminate_zeros()
    abo.sum_duplicates()
    abo.sort_indices()
    print "t4",time.time()-t0    
    # now determine what contributes to each entry of abo.
    # first create an index matrix for each of a and b
    # dense matrices are much faster look-ups than sparse
    ai = csr_matrix((range(1,len(a.data)+1), a.indices, a.indptr), dtype=int32).todense()
    bi = csr_matrix((range(1,len(b.data)+1), b.indices, b.indptr), dtype=int32).todense()
    print "t5",time.time()-t0    
    
    # determine the row, col pairs that we need to take products of
    abop = zip(abo.indptr[:-1], abo.indptr[1:])
    abij = [(i,j) for i,p in enumerate(abop) for j in abo.indices[p[0]:p[1]]]
    print "t5.5",time.time()-t0  
    ap = zip(a.indptr[:-1], a.indptr[1:])  
#    arind = [ai.getrow(i).indices for i in range(len(a.indptr)-1)]
    arind = [a.indices[s:t] for s,t in ap]
    print "t6",time.time()-t0    
    add = numpy.matrix.__add__
    data = [reduce(add,[prod(a.data[ai[i,k]-1], b.data[bi[k,j]-1]) for k in arind[i] if bi[k,j] > 0 ]) for i,j in abij]
    
    
#    arbc = [(ai.getrow(i),bi.getcol(j)) for i,p in enumerate(abop) for j in abo.indices[p[0]:p[1]]]
#    print len(arbc)
#    print "t6",time.time()-t0    
#    
##    print b.data.shape
##    print numpy.sum([b.data[k-1] for k in bi.data], axis=0)
#    print "t6.5",time.time()-t0
#    # take products
#    data = [numpy.sum([prod(a.data[ar[0,k]-1], b.data[bc[k,0]-1]) for k in ar.indices if bc[k,0] > 0], axis=0) for ar,bc in arbc]
    print "t7",time.time()-t0    
        
    return bsr_matrix((numpy.array(data), abo.indices, abo.indptr))    

@print_timing
def collapsebsr(c, b):
    """ Sums a sparse block matrix across common faces
    
    c should be a list of lists of faces to collapse.  
    b is a block for each pair of uncollapsed faces
    """
    from scipy.sparse import csr_matrix, bsr_matrix
    from scipy import int32
        
    # first lets determine the new sparsity structure
    C = csr_matrix((numpy.ones(len(numpy.hstack(c))), numpy.hstack(c), range(0,len(c)+1)))
    B = csr_matrix((numpy.ones(len(b.indices)), b.indices, b.indptr ))
    CBCt = (C * B * C.transpose()).tocsr()
    CBCt.sum_duplicates()
    CBCt.sort_indices()
    print CBCt.indices.shape
    
    #Now lets create an index matrix into B
    Bi = csr_matrix((range(1,len(b.indices)+1), b.indices, b.indptr),dtype=scipy.int32)
    # Extract the list of lists of b.data that need to be summed
    bdata = [[b.data[Bi[ii,jj]-1] for ii in i for jj in j if Bi[ii,jj] > 0] for i in c for j in c ]
    print [len(bs) for bs in bdata]
    # Sum all the blocks for each entry.  
    data = numpy.array([numpy.sum(bs,axis=0) for bs in bdata])
    print data.shape
                   
    return bsr_matrix((data, CBCt.indices, CBCt.indptr))    

class innerproduct(object):
    """ A class to calculate inner products based on vandermonde matrices """
    def __init__(self, vandermondes, quadweights):
        """ vandermondes is a list of vandermonde matrices
            quadweights is a list of quadrature weights """  
        self.v = vandermondes
        self.w = quadweights
    
    def product(self, i,j):
        """ Return the inner product of the ith face against the jth face """
    # It should be the case that w[i] = w[j], otherwise the
    # inner product makes no sense.
        return numpy.dot(numpy.multiply(self.v[i].H,self.w[i].A.flatten()), self.v[j])
    

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