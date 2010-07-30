'''
Created on Jul 15, 2010

@author: joel
'''
import unittest

class testSparse(unittest.TestCase):
    
    def testToCSR(self):
        from PyPWDG.Utils.sparse import vbsr_matrix
        from numpy import mat, bmat, zeros     
        blocks = [mat([[1,2],[3,4]]), mat([[5],[6]]), mat([[7],[8],[9]])]
        indices = [0,1,1]
        indptr = [0,2,3]
        bsizei = [2,3]
        bsizej = [2,1]
        vbsr = vbsr_matrix(blocks,indices,indptr,bsizei,bsizej)
        csr = vbsr.tocsr()
                
        self.assertEquals(csr.get_shape(), (5,3))
        truemat = bmat([[blocks[0], blocks[1]],[zeros((3,2)), blocks[2]]])
        self.assertTrue((csr.todense() == truemat).all())
                       