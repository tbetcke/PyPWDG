'''
Created on May 12, 2010

@author: joel
'''
import unittest
from dg3d import *
import numpy
from numpy import array as na

class TestDG3d(unittest.TestCase):


    def setUp(self):
        self.mesh = cubemesh()


    def tearDown(self):
        pass


    def testMesh(self):
        printMesh(self.mesh._mesh)
    
    def testFToVs(self):
        print "Our Faces:"
        ftovs = self.mesh.ftovs
        for i,v in enumerate(ftovs): print i,v

    def testEToF(self):
        print "Element to Face association"
        # check that each element has 4 faces
        for fs in self.mesh.etof: self.assertEqual(len(fs),4)
            
        
    def testNormal(self):
        points = numpy.array([[0,0,0],[1,0,0],[1,1,0]])
        n = normalM(points)
        map(self.assertEqual, n, [0,0,1])
    
    def testFacePoints(self):
        print tetmesh().facePoints(na([(0.2,0.2),(0,0)]))

    def testQuadPoints(self):
        for n in range(1,10):
            [x,w] = quadPoints(n)
            self.assertAlmostEqual(sum(w),0.5) # \int 1
            self.assertAlmostEqual(x[:,0].transpose() * w, 1.0/6) # \int x
            self.assertAlmostEqual(x[:,1].transpose() * w, 1.0/6) # \int y
            self.assertAlmostEqual((1-x[:,0].transpose()) * w, 1.0/3) # \int 1-x

    def testBoundaryAndConnectivity(self):
        m = cubemesh()
        ff = numpy.hstack(m.etof)
        self.assertTrue(len(set(m.average.indices).intersection(set(m.boundary.indices)) ) == 0)
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testMesh']
    unittest.main()