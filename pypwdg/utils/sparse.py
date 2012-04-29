'''
Created on Jul 14, 2010

@author: joel
'''
from pypwdg.utils.timing import print_timing, Timer
import numpy as np
import scipy.sparse as ss

def createvbsr(mat, blocks, bsizerows = None, bsizecols = None):
    """ Creates a variable block sparse matrix
    
    mat gives the sparsity pattern.  
    blocks is a callable returning the (i,j) block
    bsizerows/cols give the block sizes.  These are optional - by default they are inferred from blocks(.,.)
    However, if mat has any empty rows or columns, the relevant bsize* entry will be zero.  This might not be
    what you want.  
    
    The (i,j)th block of the vbsr is mat[i,j] * blocks(i,j)
    """
    csr = mat.tocsr()
    csr.sum_duplicates()
    csr.eliminate_zeros()
    csr.sort_indices()
    
    zipip = zip(csr.indptr[:-1], csr.indptr[1:])
    coords = [(i,j) for i,p in enumerate(zipip) for j in csr.indices[p[0]:p[1]] ]
    data = [csr.data[n] * blocks(i,j) for n, (i,j) in enumerate(coords)]
    s = csr.get_shape()
    if bsizerows is None: bsizerows = [None]*s[0]
    if bsizecols is None: bsizecols = [None]*s[1]
    for (i,j), block in zip(coords, data):
        
        (r,c) = block.shape
#        print r,c
        if not bsizerows[i] in [r, None]: raise ValueError("Incompatible block sizes. row:%s, r:%s, bsizerows[i]:%s" %(i,r,bsizerows[i]))  
        if not bsizecols[j] in [c, None]: raise ValueError("Incompatible block sizes. col:%s, c:%s, bsizecols[i]:%s %s" %(j,c,bsizecols[i]))
        bsizerows[i] = r
        bsizecols[j] = c 
    
    bsizerows = [0 if x is None else x for x in bsizerows]
    bsizecols = [0 if x is None else x for x in bsizecols]
        
    return vbsr_matrix(data, csr.indices, csr.indptr, bsizerows, bsizecols)

class vbsr_matrix(object):
    '''
    A block-sparse matrix containing blocks of variable size.
    
    The blocks must be dense matrices at the moment, although this would be easy to generalise
    
    A certain amount of consistency checking is done one the block sizes for the various operators.  However
    this should not be relied upon yet.  This is an incomplete class - I'm just adding functionality as I need it  
    
    @see: test.pypwdg.Utils

    '''


    def __init__(self, blocks, indices, indptr, bsizei, bsizej, scalar = 1.0):
        import numpy
        '''
        blocks, indices, indptr: csr sparse representation of matrix in terms of subblocks
        bsizei, bsizej: arrays giving the notional size of each subblock
        
        Obviously, blocks[k] should have size (bsize[i], bsizej[indices[k]]) for k in indptr[i]:indptr[i+1],
        however, things may still work if that condition is not satisfied.  The reason that we don't just
        infer bsizei and bsizej from blocks is to allow for empty (blocks of) rows and columns
         
        '''
        self.blocks = blocks
        self.indices = numpy.array(indices, dtype=int)
        self.indptr = numpy.array(indptr, dtype=int)    
        self.bsizei = numpy.array(bsizei, dtype=int)
        self.bsizej = numpy.array(bsizej, dtype=int)
        self.bindj = numpy.concatenate(([0],bsizej)).cumsum()
        self.bindi = numpy.concatenate(([0],bsizei)).cumsum()
        self.scalar = scalar
#        self.shape = (sum(self.bsizei), sum(self.bsizej))
        self.shape = (self.bindi[-1], self.bindj[-1])
        

    def nnz(self):
        '''Return number of nonzeros in blocks'''
        nelems=0
        for b in self.blocks: nelems+=b.shape[0]*b.shape[1]
        return nelems
    
    def tocsr(self):
        ''' Return this matrix in csr format '''
        csrdata = []
        csrind = []
        rowlengths = np.zeros(self.shape[0],dtype=int)
        
        for i, (p0, p1) in enumerate(zip(self.indptr[:-1], self.indptr[1:])):
            if p1 > p0:
                csrdata.append(np.hstack(self.blocks[p0:p1]).ravel())
                rowinds = np.concatenate([np.arange(self.bindj[idx], self.bindj[idx + 1],dtype=int) for idx in self.indices[p0:p1]])
                csrind.append(np.tile(rowinds, [self.bsizei[i], 1]).ravel())
                rowlengths[self.bindi[i]:self.bindi[i + 1]] = len(rowinds)
            else:
                rowlengths[self.bindi[i]:self.bindi[i + 1]] = 0
        if len(csrind):
            csrptr = np.concatenate(([0], rowlengths)).cumsum()
            naind = np.concatenate(csrind)
            d = np.concatenate(csrdata) * self.scalar
            csr = ss.csr_matrix((d, naind, csrptr), shape=self.shape)
        else:
            csr = ss.csr_matrix(self.shape)
        return csr

                 
    def tocsr_old(self):
        ''' Return this matrix in csr format '''
        from numpy import concatenate, array
        from scipy.sparse import csr_matrix
        bj = zip(self.blocks, self.indices)
        csrdata = []
        csrptr = [0]
        csrind = []
        cptr = 0
        for i,(p0,p1) in enumerate(zip(self.indptr[:-1], self.indptr[1:])):
            for ii in range(self.bsizei[i]):
                for b,j in bj[p0:p1]:
                    csrdata.append(b[ii,:].ravel() * self.scalar)
                    csrind.append(range(b.shape[1])+self.bindj[j])
                    cptr += b.shape[1]
                csrptr.append(cptr)
        naind = array(concatenate(csrind),dtype=int) # for some reason, get a warning about non-ints from sparse if don't make this explicit
        return csr_matrix((concatenate(csrdata), naind, csrptr), shape=(len(csrptr)-1, self.bindj[-1]))
    
    def todense(self):
        return self.tocsr().todense()
    
    def subrows(self, structurerows):
        ''' Return the rows in the csr matrix associated with structurerows'''
        rowidxs = []
        for s, i in zip(self.bsizei[structurerows], self.bindi[structurerows]):
            rowidxs.append(np.arange(i, i+s))
        return np.concatenate(rowidxs) if len(rowidxs) else np.array([])
        
    
    def _mul(self, lindices, lindptr, ldata, lshape, lsizes, rindices, rindptr, rdata, rshape, rsizes, otherscalar, prod):
        """ `multiply' a block matrix by a sparse matrix 
        
        l... and r... should be the data for (variable block) sparse matrices with compatible sparsity structures.
        At least one should be a block sparse matrix.  Effectively, we replace each entry x_ij of the non-block matrix, 
        x by a_ij I where I is an appropriately sized identity matrix; and then perform the multiplication.
        
        *indices, *indptr, *data are the standard csr sparse data structures.  
        *shape is the shape at the structure level
        *sizes are the sizes of the blocks that aren't collapsed by the multiplication (i.e. the number of rows for
        the LHS and cols for the RHS).  If the block structure is incompatible with the multiplication, an exception 
        is thrown.  If *sizes consists of -1 entries, it will be guessed by the routine. 
        
        you could make prod something like lambda a,b:a if you know that all the entries in the non-block are 1
        although it doesn't seem to make much difference
                
        As a side effect, if both a and b are block sparse then it might perform a block by block sparse matrix multiply.  There
        is no reason to expect this to be more efficient than the inbuilt multiply.  However, we now have
        access to the prod function, which could therefore be distributed.  In any case, if a and b are both
        block sparse then prod should be set to something like numpy.dot, otherwise, the multiplication
        will be elementwise for each block multiply.  Note also that in this case, it's likely to be 
        easier just to use createblock
        
        as much as possible, the standard sparse matrix routines are used to do all the work on the underlying
        sparsity structure.  this might look a bit inefficient, but since all that's done in C++, it's not.
        Caveat refactorer - it's very easy to slow this down.
        """        
        from numpy import ones
        from scipy.sparse import csr_matrix
        from scipy import int32
        
        # for each element of the product, we will iterate through a column of b.
        # it's much more efficient to do this if b is in column format.  
        # since b might be a block matrix and there's no blockcsr, we can't convert it
        # directly.  instead we build an index matrix:    
        ri = csr_matrix((range(0,len(rdata)), rindices, rindptr), dtype=int32, shape = rshape).tocsc()
        ri.sort_indices()
        
        # now determine the block-level sparsity structure of a . b
        lo = csr_matrix((ones(len(ldata)), lindices, lindptr), shape = lshape)
        ro = csr_matrix((ones(len(rdata)), rindices, rindptr), shape = rshape)
        lro = lo * ro
        lro.sum_duplicates()
        lro.eliminate_zeros()
        lro.sort_indices()
        
        # the data for the new matrix
        data = []
        indices = []
        indptr = [0]
                
        # iterate through each (i,j) element in abo
        for i, abop in enumerate(zip(lro.indptr[:-1], lro.indptr[1:])):
            for j in lro.indices[abop[0]:abop[1]]:
                # multiply the ith row of a by the jth column of b
                [pa,pa1] = lindptr[i:i+2]
                [pb,pb1] = ri.indptr[j:j+2]
                ab = None
                while (pa < pa1 and pb < pb1):
                    ak = lindices[pa]
                    bk = ri.indices[pb]
                    if ak==bk: 
                        pab = prod(ldata[pa], rdata[ri.data[pb]])
                        if ab is None: ab = pab
                        else: ab = ab + pab
                    if ak <= bk: pa+=1
                    if bk <= ak: pb+=1
                if (lsizes[i], rsizes[j]) != ab.shape: raise ValueError("Incompatible block sizes %s, %s"%((lsizes[i], rsizes[j]), ab.shape)) 
                data.append(ab * otherscalar * self.scalar) 
                indices.append(j)                
            indptr.append(len(indices))
         
        return vbsr_matrix(data, indices, indptr, lsizes, rsizes) 
    
    def _scalarmul(self, x):
        return vbsr_matrix(self.blocks, self.indices, self.indptr, self.bsizei, self.bsizej, self.scalar * x)
        
    def _mulstructarray(self, x):
        """ Multiply by a dense structure vector
        
        In theory it should be easy to multiply by a dense structure matrix - just need to work out the correct
        reshapes and sums """
        from numpy import newaxis, vstack, zeros,sum
        data = []
        for i, (ip0,ip1) in enumerate(zip(self.indptr[:-1], self.indptr[1:])):
            idx = self.indices[ip0:ip1]
            
            if len(idx) > 0 : data.append(sum(self.blocks[ip0:ip1] * x.reshape(-1,1,1)[idx], axis=0) )
            else:
                data.append(zeros((self.bsizei[i], self.bsizej[0]))) 
        ret = vstack(data) * self.scalar
        return ret
    
    # This appears to be a bad idea.  It's cheaper just to use .tocsr().  
    def matmat(self, x):
        from numpy import dot, sum, zeros, vstack
        if len(x.shape) == 1: x = x.reshape(-1,1)
        data = []
        xslices = [x[b0:b1] for b0,b1 in zip(self.bindj[:-1], self.bindj[1:])]
        for i, (ip0,ip1) in enumerate(zip(self.indptr[:-1], self.indptr[1:])):
            idx = self.indices[ip0:ip1]                        
            if len(idx) > 0 :
                data.append(sum([dot(b, xslices[ii]).reshape(-1,x.shape[1]) for b,ii in zip(self.blocks[ip0:ip1], idx)], axis=0) )
            else: data.append(zeros((self.bsizei[i], x.shape[1]))) 

        return vstack(data)    
            
        
    def _calculatesizes(self, csr, bsize):
        sizes = [bsize[csr.indices[i0:i1]] for i0,i1 in zip(csr.indptr[:-1], csr.indptr[1:])]
        nonempty = filter(len, sizes)
        minsizes = map(min, nonempty)
        maxsizes = map(max, nonempty)
        if minsizes != maxsizes: raise ValueError("Incompatible block sizes")
        asizes = np.zeros((len(sizes)))
        asizes[np.array(map(len, sizes)) > 0] = maxsizes
        return asizes 

    def __mul__(self, other):
        """ Multiply this variable block sparse matrix by a sparse matrix at the structure level
        
        Doesn't yet cope with vbsr * vbsr.
        """         
        import numpy
        from scipy.sparse import issparse
        if numpy.isscalar(other): return self._scalarmul(other)
        if issparse(other): 
            rcsr = other.tocsr()
            colsizes = self._calculatesizes(other.transpose().tocsr(), self.bsizej) 
            
            return self._mul(self.indices, self.indptr, self.blocks, (len(self.bsizei), len(self.bsizej)), self.bsizei, \
                             rcsr.indices, rcsr.indptr, rcsr.data, rcsr.get_shape(), colsizes, 1.0, numpy.multiply)
        if (isinstance(other, numpy.ndarray) and len(other.shape) == 1): return self._mulstructarray(other)
        return NotImplemented
        
    def __rmul__(self, other):
        """ This doesn't work using the * operator because the sparse matrix classes don't return 
        NotImplemented from __mul__ for an unknown rhs, annoyingly"""
        import numpy
        from scipy.sparse import issparse
        if numpy.isscalar(other): return self._scalarmul(other)
        if not issparse(other): return NotImplemented
        lcsr = other.tocsr()
        lcsr.eliminate_zeros()
        lcsr.sort_indices()
        rowsizes = self._calculatesizes(lcsr, self.bsizei)
        return self._mul(lcsr.indices, lcsr.indptr, lcsr.data, lcsr.get_shape(), rowsizes, \
                         self.indices, self.indptr, self.blocks, (len(self.bsizei), len(self.bsizej)), self.bsizej, 1.0,  numpy.multiply)
    
    # this makes sum work for vbsr matrices
    def __radd__(self, other):
        if other==0: return self
        return NotImplemented
    
#    @print_timing    
    def __add__(self, other):
        from numpy import array_equal
        if other==0: return self
        sizeinonzero = np.logical_and(other.bsizei != 0, self.bsizei !=0) 
        if not array_equal(other.bsizei[sizeinonzero], self.bsizei[sizeinonzero]):
            print other.bsizei[sizeinonzero], self.bsizei[sizeinonzero] 
            raise ValueError("Incompatible block sizes")
        sizejnonzero = np.logical_and(other.bsizej != 0, self.bsizej !=0) 
        if not array_equal(other.bsizej[sizejnonzero], self.bsizej[sizejnonzero]): raise ValueError("Incompatible block sizes")
        
        blocks = []
        indices = []
        indptr = [0]
        n = len(self.bsizej)
        # Iterate through rows of both matrices simultaneously
        for (ap0, ap1),(bp0,bp1) in zip(zip(self.indptr[:-1], self.indptr[1:]), zip(other.indptr[:-1], other.indptr[1:])):                        
            # Iterate through columns simultaneously
            while(ap0 < ap1 or bp0 < bp1):
                # Check whether we've reached the end of a row in one matrix
                aj = n if ap0 == ap1 else self.indices[ap0]
                bj = n if bp0 == bp1 else other.indices[bp0]
                j = min(aj,bj)
                if aj < bj:
                    ab = self.blocks[ap0] * self.scalar
                    ap0+=1
                elif aj > bj:
                    ab = other.blocks[bp0] * other.scalar
                    bp0+=1
                elif aj == bj:
                    ab = self.blocks[ap0]* self.scalar + other.blocks[bp0] * other.scalar
                    ap0+=1
                    bp0+=1
                
                indices.append(j)
                blocks.append(ab)
            indptr.append(len(indices))
#        print "sparse.add ", np.sum(self.blocks), np.sum(other.blocks), np.sum(blocks)
        return vbsr_matrix(blocks, indices, indptr, np.amax((self.bsizei, other.bsizei), axis=0), np.amax((self.bsizej,other.bsizej),axis=0))
    
    def __sub__(self, other):
        return self.__add__(-other)
    
    def __neg__(self):
        return -1.0 * self    
        
        
