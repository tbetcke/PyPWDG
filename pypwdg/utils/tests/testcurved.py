'''
Created on Sep 1, 2011

@author: joel
'''
import unittest
import numpy as np
import pypwdg.utils.curved as puc

class Test(unittest.TestCase):


    def testCurve1(self):
        ''' Map the face x=0 to a parabola '''
        n = 2
        vertices = np.vstack((np.eye(n), np.zeros((1,n))))
        subsimplexids = np.array([1,2]) 
        surface = lambda t: np.hstack((-t*(1-t)/4, t))
        fm = puc.CurvedMapping(vertices, subsimplexids, surface)
        np.testing.assert_almost_equal(vertices, fm.apply(vertices))
        N = 10
        yaxis = np.zeros((10,2))
        yaxis[:,1] = np.linspace(0,1,N)
        mapyaxis = fm.apply(yaxis)
        print mapyaxis
        np.testing.assert_almost_equal(-mapyaxis[:,1]*(1-mapyaxis[:,1])/4, mapyaxis[:,0])
        
    def testCurve2(self):
        ''' Map the face x=0 to a parabola that does not intersect the original face at its vertices '''
        n = 2
        vertices = np.vstack((np.eye(n), np.zeros((1,n))))
        subsimplexids = np.array([1,2]) 
        surface = lambda t: np.hstack((-t*(1-t)/4-1, t))
        fm = puc.CurvedMapping(vertices, subsimplexids, surface)
        N = 10
        yaxis = np.zeros((10,2))
        yaxis[:,1] = np.linspace(0,1,N)
        mapyaxis = fm.apply(yaxis)
        print mapyaxis
        np.testing.assert_almost_equal(-mapyaxis[:,1]*(1-mapyaxis[:,1])/4-1, mapyaxis[:,0])

    def testCurve3(self):
        ''' This time we're just mapping an edge to the parabola '''
        vertices = np.array([[0,0],[0,1]])
        subsimplexids = np.array([0,1])
        surface = lambda t: np.hstack((-t*(1-t)/4-1, t))
        fm = puc.CurvedMapping(vertices, subsimplexids, surface)
        
        N = 10
        yaxis = np.zeros((10,2))
        yaxis[:,1] = np.linspace(0,1,N)
        mapyaxis = fm.apply(yaxis)
        print mapyaxis
        np.testing.assert_almost_equal(-mapyaxis[:,1]*(1-mapyaxis[:,1])/4-1, mapyaxis[:,0])
        
        
    def testVertexMap(self):
        ''' Test that just mapping one vertex works '''
        n = 2
        vertices = np.vstack((np.eye(n), np.zeros((1,n))))
        subsimplexids = np.array([1])
        surface = lambda t: np.hstack(((1-t)**2+1,t)) # a parabola with its apex at 1,1, which is the closet point to (0,1)
        fm = puc.CurvedMapping(vertices, subsimplexids, surface)
        N = 5
        points = puc.uniformreferencepoints(2, N)
        mpoints = fm.apply(points)
        print points
        print mpoints
        M = np.array([[1,0],[1,1]]) # the desired map should be a shear
        np.testing.assert_almost_equal(mpoints, np.dot(points, M))

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testCurve1']
    unittest.main()