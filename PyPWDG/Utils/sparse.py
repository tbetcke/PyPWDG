'''
Created on Jul 14, 2010

@author: joel
'''

class vbsr_matrix(object):
    '''
    A block-sparse matrix containing blocks of variable size
    '''


    def __init__(self, blocks, indices, indptr, bsizei, bsizej):
        import numpy
        '''
        blocks, indices, indptr: csr sparse representation of matrix in terms of subblocks
        bsizei, bsizej: arrays of tuples giving the notional size of each subblock
        
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
            
        
    def tocsr(self):
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
                    csrdata.append(b[ii,:].A.flatten())
                    csrind.append(range(b.shape[1])+self.bindj[j])
                    cptr += b.shape[1]
                csrptr.append(cptr)
        return csr_matrix((concatenate(csrdata), concatenate(csrind), csrptr), shape=(len(csrptr)-1, self.bindj[-1]))
    
    def __mul__(self, other):
        