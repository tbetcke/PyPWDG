'''
Created on Jul 15, 2010

@author: joel
'''
import unittest
from pypwdg.utils.sparse import vbsr_matrix
from pypwdg.utils.quadrature import *
from test.pypwdg import __path__ as path

import numpy as np
from numpy import mat, bmat, zeros, equal  
import math   

class testSparse(unittest.TestCase):
#todo: create some tests for vbsrs with empty rows and columns

    def setUp(self):
        # VS1 is an example variable size block matrix
        blocks = [mat([[1,2],[3,4]]), mat([[1],[2]]), mat([[1],[2],[3]])]
        indices = numpy.array([0,1,1], dtype=int)
        indptr = numpy.array([0,2,3], dtype=int)
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
        indices = numpy.array([0,1], dtype=int)
        indptr = numpy.array([0,1,2],dtype=int)
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
        
    def testScalarMulSubAndNeg(self):
        m1 = self.VS1 * -4.0
        m2 = -4.0 * self.VS1
        m3 = 4.0 * (-self.VS1)
        m4 = m1 + m2 - m3
        self.assertTrue(np.array_equal(m1.tocsr().todense(), self.VS1D*-4.0))
        self.assertTrue(np.array_equal(m2.tocsr().todense(), self.VS1D*-4.0))
        self.assertTrue(np.array_equal(m3.tocsr().todense(), self.VS1D*-4.0))
        self.assertTrue(np.array_equal(m4.tocsr().todense(), self.VS1D*-4.0))
    
    def testVectorMul(self):
        blocks = [mat([[1],[2]]), mat([[3],[4]]), mat([[1],[2],[3]])]
        indices = numpy.array([0,1,1], dtype=int)
        indptr = numpy.array([0,2,2,3], dtype=int)
        bsizei = [2,2,3]
        bsizej = [1,1]
        M = vbsr_matrix(blocks,indices,indptr,bsizei,bsizej)
        v = M * numpy.array([1,2])
        self.assertTrue(np.array_equal(v, M.tocsr().todense() * mat([1,2]).transpose()))
        
        
class testQuadrature(unittest.TestCase):
    
    def testTriangle(self):
        for n in range(1,10):
            x,w = trianglequadrature(n)
            # test integration of constants
            self.assertAlmostEqual(sum(w), 0.5)
            # integrate x
            self.assertAlmostEqual(numpy.dot(x[:,0],w), 1.0/6)
            # integrate y
            self.assertAlmostEqual(numpy.dot(x[:,1],w), 1.0/6)

    def testLegendre(self):
        for n in range(1,10):
            x,w = legendrequadrature(n)
            for m in range(0,2*n):
                # test integration of x^m
                self.assertAlmostEqual(np.dot(x.transpose() ** m, w), 1.0/(m+1) )
                
            
class testGeometry(unittest.TestCase):
    
    def setUp(self):
        from pypwdg.mesh.gmsh_reader import gmsh_reader
        from pypwdg.mesh.mesh import Mesh
        
        mesh_dict=gmsh_reader(path[0] + '/../../examples/2D/square.msh')
        self.squaremesh=Mesh(mesh_dict,dim=2)
        mesh_dict=gmsh_reader(path[0] + '/../../examples/3D/cube_coarse.msh')
        self.cubemesh=Mesh(mesh_dict,dim=3)
    
    def testPointsToElement(self):
        from pypwdg.utils.geometry import pointsToElement
        from pypwdg.mesh.structure import StructureMatrices
        
        points = numpy.array([[0.2,0.1],[0.3,0.4], [0.5,0.6]])
        pe = pointsToElement(points, self.squaremesh, StructureMatrices(self.squaremesh))
        
        