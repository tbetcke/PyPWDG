'''
Created on Jul 14, 2010

@author: joel
'''

class vbsr_matrix(object):
    '''
    A block-sparse matrix containing blocks of variable size.
    
    The blocks must be dense matrices at the moment, although this would be easy to generalise
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
        self.indices = indices
        self.indptr = indptr    
        self.bsizei = bsizei
        self.bsizej = bsizej
        self.bindj = numpy.concatenate(([0],bsizej)).cumsum()
        self.scalar = scalar
            
        
    def tocsr(self):
        ''' Return this matrix in csr format '''
        from numpy import concatenate
        from scipy.sparse import csr_matrix
        bj = zip(self.blocks, self.indices)
        csrdata = []
        csrptr = [0]
        csrind = []
        cptr = 0
        for i,(p0,p1) in enumerate(zip(self.indptr[:-1], self.indptr[1:])):
            for ii in range(self.bsizei[i]):
                for b,j in bj[p0:p1]:
                    csrdata.append(b[ii,:].A.flatten() * self.scalar)
                    csrind.append(range(b.shape[1])+self.bindj[j])
                    cptr += b.shape[1]
                csrptr.append(cptr)
        return csr_matrix((concatenate(csrdata), concatenate(csrind), csrptr), shape=(len(csrptr)-1, self.bindj[-1]))
    
    def _mul(self, lindices, lindptr, ldata, lshape, lsizes, rindices, rindptr, rdata, rshape, rsizes, prod):
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
        lro.eliminate_zeros()
        lro.sum_duplicates()
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
                (ln,rn) = ab.shape
                if lsizes[i] == -1: lsizes[i] = ln
                elif lsizes[i] != ln: raise ValueError("Incompatible block sizes") 
                if rsizes[j] == -1: rsizes[j] = rn
                elif rsizes[j] != rn: raise ValueError("Incompatible block sizes") 
                data.append(ab) 
                indices.append(j)                
            indptr.append(len(indices))
         
         
        return vbsr_matrix(data, indices, indptr, lsizes, rsizes) 
    
    def _scalarmul(self, x):
        pass
        
        
    def __mul__(self, other):
        """ Multiply this variable block sparse matrix by a sparse matrix at the structure level
        
        Doesn't yet cope with variable * variable.
        """         
        import numpy
        from scipy.sparse import issparse
        if numpy.isscalar(other): return self._scalarmul(other)
        if not issparse(other): return NotImplemented
        rcsr = other.tocsr()
        return self._mul(self.indices, self.indptr, self.blocks, (len(self.bsizei), len(self.bsizej)), self.bsizei, rcsr.indices, rcsr.indptr, rcsr.data, rcsr.get_shape(), numpy.ones(rcsr.get_shape()[1])*-1, numpy.multiply)
        
    def __rmul__(self, other):
        """ This doesn't work using the * operator because the sparse matrix classes don't return 
        NotImplemented from __mul__ for an unknown rhs, annoyingly"""
        import numpy
        from scipy.sparse import issparse
        if numpy.isscalar(other): return self._scalarmul(other)
        if not issparse(other): return NotImplemented
        lcsr = other.tocsr()
        return self._mul(lcsr.indices, lcsr.indptr, lcsr.data, lcsr.get_shape(), numpy.ones(lcsr.get_shape()[0])*-1, self.indices, self.indptr, self.blocks, (len(self.bsizei), len(self.bsizej)), self.bsizej, numpy.multiply)
    
    def __add__(self, other):
        print self.bsizei
        print other.bsizei
        if not other.bsizei == self.bsizei: raise ValueError("Incompatible block sizes")
        if not other.bsizej == self.bsizej: raise ValueError("Incompatible block sizes")
        
        blocks = []
        indices = []
        indptr = [0]
        n = len(self.bsizej)
        # Iterate through rows of both matrices simultaneously
        for (ap0, ap1),(bp0,bp1) in zip(zip(self.indptr[:-1], self.indptr[1:]), zip(other.indptr[:-1], other.indptr[1:])):                        
            # Iterate through columns simultaneously
            while(ap0 < ap1 or bp0 < bp1):
                aj = self.indices[ap0]
                bj = self.indices[bp0]
                # Check whether we've reached the end of a row in one matrix
                if ap0 == ap1: aj = n
                if bp0 == bp1: bj = n
                j = min(aj,bj)
                if aj < bj:
                    ab = self.blocks[ap0]
                    ap0+=1
                elif aj > bj:
                    ab = other.blocks[bp0]
                    bp0+=1
                elif aj == bj:
                    ab = self.blocks[ap0] + other.blocks[bp0]
                    ap0+=1
                    bp0+=1
                
                indices.append(j)
                blocks.append(ab)
            indptr.append(len(indices))
        
        return vbsr_matrix(blocks, indices, indptr, self.bsizei, self.bsizej)
    
        
        
        