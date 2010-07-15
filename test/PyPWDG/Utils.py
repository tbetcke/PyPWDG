'''
Created on Jul 15, 2010

@author: joel
'''
import unittest

class testSparse(unittest.TestCase):
    
    def testToCSR(self):
        from PyPWDG.Utils.sparse import vbsr_matrix
        from numpy import mat     
        blocks = [mat([[1,2],[3,4]]), mat([[1],[2]]), mat([[1],[2],[3]])]
        indices = [0,1,1]
        indptr = [0,2,3]
        bsizei = [2,3]
        bsizej = [2,1]
        vbsr = vbsr_matrix(blocks,indices,indptr,bsizei,bsizej)
        csr = vbsr.tocsr()
        
        print csr.todense()
        self.assertEquals(csr.get_shape(), (5,3))