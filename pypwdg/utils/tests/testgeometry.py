'''
Created on Dec 19, 2010

@author: joel
'''
import unittest

import pypwdg.utils.geometry as pug
import test.utils.mesh as tum

import numpy as np

class TestPointsToElement(unittest.TestCase):


    def testPToE3D(self):        
        self.compareStructuredAndUnstructured(tum.meshes3d(), 3)

    def testPToE2D(self):
        self.compareStructuredAndUnstructured(tum.meshes2d(), 2)

    def compareStructuredAndUnstructured(self, meshes, dim):        
        """ Compare the structured and unstructured element-to-point routines
        
            Since the unstructured routine picks an arbitrary element for points that lie on boundaries
            but the structured routine will say that such points lie in both elements, this routine
            can at best say that the two routines are compatible.
            """
        vertices = np.array([[0]*dim,[1]*dim])
        sp = pug.StructuredPoints(vertices, np.array([5]*dim))
        for mesh in meshes:
            points = sp.getPoints(vertices)[1]
#            print points
            etopunstructured = pug.pointsToElement(points, mesh)
            pointscheck = (etopunstructured==-1)
            
            for e in range(mesh.nelements):
                ps =  pug.elementToStructuredPoints(sp, mesh, e)[0]
                self.assertFalse((etopunstructured[ps]==-1).any())
                pointscheck[ps] |= (etopunstructured[ps]==e)
            
            self.assertTrue(pointscheck.all())
