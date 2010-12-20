'''
Created on Dec 19, 2010

@author: joel
'''
import pypwdg.utils.sparse as pus

import unittest

import numpy as np

class testSparse(unittest.TestCase):
#todo: create some tests for vbsrs with empty rows and columns

    def setUp(self):
        # VS1 is an example variable size block matrix
        blocks = [np.array([[1,2],[3,4]]), np.array([[1],[2]]), np.array([[1],[2],[3]])]
        indices = np.array([0,1,1], dtype=int)
        indptr = np.array([0,2,3], dtype=int)
        bsizei = [2,3]
        bsizej = [2,1]
        self.VS1 = pus.vbsr_matrix(blocks,indices,indptr,bsizei,bsizej)
        self.VS1D = self.VS1.tocsr().todense()

    
    def tearDown(self):
        pass

    
    def testToCSR(self):
        csr = self.VS1.tocsr()
        self.assertEquals(csr.get_shape(), (5,3))
        self.assertTrue((self.VS1D == csr.todense()).all())
        
    def testMul(self):
        from scipy.sparse import eye, csr_matrix
        block = np.array([[1.0,2],[3,4]])
        indices = np.array([0,1], dtype=int)
        indptr = np.array([0,1,2],dtype=int)
        bsizei = [2,2]
        bsizej = [2,2]
        # Construct a 2x2 block matrix of 2x2 blocks
        b = pus.vbsr_matrix([block, block], indices, indptr, bsizei, bsizej)
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
        x = csr_matrix(np.mat([[1],[1]]))
        bx = b * x
        self.assertTrue((bx.tocsr().todense() == np.bmat([[block], [block]])).all())
        
        # Check that this won't work on the matrix with variable (and incompatible) block sizes
        self.assertRaises(ValueError, lambda : self.VS1 * x)
        
        #Sum some columns:
        xb = b.__rmul__(x.transpose())
        self.assertTrue((xb.tocsr().todense() == np.bmat([[block, block]])).all())
        
    def testAdd(self):
        m = self.VS1 + self.VS1
        self.assertTrue((m.tocsr().todense() == self.VS1D * 2).all())
        
    def testScalarMulSubAndNeg(self):
        m1 = self.VS1 * -4.0
        m2 = -4.0 * self.VS1
        m3 = 4.0 * (-self.VS1)
        m4 = m1 + m2 - m3
        self.assertTrue(np.array_equal(m1.tocsr().todense(), self.VS1D*-4.0))
        self.assertTrue(np.array_equal(m2.tocsr().todense(), self.VS1D*-4.0))
        self.assertTrue(np.array_equal(m3.tocsr().todense(), self.VS1D*-4.0))
        self.assertTrue(np.array_equal(m4.tocsr().todense(), self.VS1D*-4.0))
    
    def testStructureVectorMul(self):
        blocks = [np.array([[1],[2]]), np.array([[3],[4]]), np.array([[1],[2],[3]])]
        indices = np.array([0,2,2], dtype=int)
        indptr = np.array([0,2,2,3], dtype=int)
        bsizei = [2,2,3]
        bsizej = [1,1,1]
        M = pus.vbsr_matrix(blocks,indices,indptr,bsizei,bsizej)
        v = M * np.array([1,3,2])
        self.assertTrue(np.array_equal(v, M.tocsr().todense() * np.mat([1,3,2]).transpose()))
        
    def testFullMul(self):
        x = np.array([1,2,3])
        xx = np.array([[1],[2],[3]])
        y = np.array([[1,4],[2,5],[3,6]])
        
        vx = self.VS1.matmat(x)
        vxx = self.VS1.matmat(xx) 
        vy = self.VS1.matmat(y)
        
        self.assertTrue(np.array_equal(vx, np.dot(self.VS1D,xx)))  
        self.assertTrue(np.array_equal(vxx, np.dot(self.VS1D,xx)))  
        self.assertTrue(np.array_equal(vy, np.dot(self.VS1D,y)))  
        