'''
Created on Jul 15, 2010

@author: joel
'''
import unittest
from PyPWDG.utils.sparse import vbsr_matrix
from numpy import mat, bmat, zeros     

class testSparse(unittest.TestCase):

    def setUp(self):
        # VS1 is an example variable size block matrix
        blocks = [mat([[1,2],[3,4]]), mat([[1],[2]]), mat([[1],[2],[3]])]
        indices = [0,1,1]
        indptr = [0,2,3]
        bsizei = [2,3]
        bsizej = [2,1]
        self.VS1 = vbsr_matrix(blocks,indices,indptr,bsizei,bsizej)
        self.VS1D = self.VS1.tocsr().todense()

    
    def tearDown(self):
        pass

    
    def testToCSR(self):
        csr = self.VS1.tocsr()
        self.assertEquals(csr.get_shape(), (5,3))
        self.assertTrue((self.VS1D == csr.todense()).all())
        
    def testMul(self):
        from scipy.sparse import eye, csr_matrix
        block = mat([[1.0,2],[3,4]])
        indices = [0,1]
        indptr = [0,1,2]
        bsizei = [2,2]
        bsizej = [2,2]
        # Construct a 2x2 block matrix of 2x2 blocks
        b = vbsr_matrix([block, block], indices, indptr, bsizei, bsizej)
        # Multiply by the identity at the block level
        bi = b * eye(2,2)
        self.assertTrue((b.tocsr().todense()==bi.tocsr().todense()).all())
        
        # Try the same things with VS1
        VS1i = self.VS1 * eye(2,2)
        self.assertTrue((self.VS1D==VS1i.tocsr().todense()).all())
        
        # Repeat the identity tests with left multiplication
        ib = b.__rmul__(eye(2,2))
        self.assertTrue((b.tocsr().todense()==ib.tocsr().todense()).all())
        
        iVS1 = self.VS1.__rmul__(eye(2,2))
        self.assertTrue((self.VS1D==iVS1.tocsr().todense()).all())
        
        #Now try summing some rows
        x = csr_matrix(mat([[1],[1]]))
        bx = b * x
        self.assertTrue((bx.tocsr().todense() == bmat([[block], [block]])).all())
        
        # Check that this won't work on the matrix with variable (and incompatible) block sizes
        self.assertRaises(ValueError, lambda : self.VS1 * x)
        
        #Sum some columns:
        xb = b.__rmul__(x.transpose())
        self.assertTrue((xb.tocsr().todense() == bmat([[block, block]])).all())
        
    def testAdd(self):
        m = self.VS1 + self.VS1
        self.assertTrue((m.tocsr().todense() == self.VS1D * 2).all())
        
