'''
Created on Aug 5, 2010

@author: joel
'''
import unittest

import PyPWDG.core
import numpy

class TestBases(unittest.TestCase):


    def testPlaneWave(self):
        from PyPWDG.core.bases import PlaneWaves
        dirs = numpy.array([[1,0]])
        k = 3
        pw = PlaneWaves(dirs, k)
        # check that there's just one function
        self.assertEqual(pw.n, 1)
        # evaluate it at 0
        fd0 = pw.values(numpy.zeros((1,2)), None)
        # answer should be 1
        self.assertAlmostEqual(fd0[0,0], 1.0)
        # now evaluate some directional derivatives
        fn0 = pw.derivs(numpy.zeros((1,2)), numpy.array([[1,0],[0,1]]))
        self.assertAlmostEqual(fn0[0,0], k*1j)
        self.assertAlmostEqual(fn0[1,0], 0)
        
class TestAssembly(unittest.TestCase):
    
    def setUp(self):
        from PyPWDG.mesh.gmsh_reader import gmsh_reader
        from PyPWDG.mesh.mesh import Mesh
        
        mesh_dict=gmsh_reader('../../examples/2D/square.msh')
        self.squaremesh=Mesh(mesh_dict,dim=2)

    
    def testLocalVandermondes(self):
        # This test is more about exercising the code than testing any results.  We need a known simple mesh to do that
        from PyPWDG.core.bases import PlaneWaves
        from PyPWDG.core.assembly import LocalVandermondes
        from PyPWDG.utils.quadrature import legendrequadrature
        dirs = numpy.array([[1,0]])
        k = 3
        pw = PlaneWaves(dirs, k)
        faceid = 0
        testelt = self.squaremesh.faces[faceid][0]
        numquads = 3
        (q,w) = legendrequadrature(numquads)
        LV = LocalVandermondes(self.squaremesh, {testelt:[pw]}, (q,w))
        v = LV.getValues(faceid)
        d = LV.getDerivs(faceid)
        self.assertEqual(v.shape, (numquads, 1))
        self.assertEqual(d.shape, (numquads, 1))
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()