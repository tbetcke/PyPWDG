'''
Created on Mar 31, 2011

@author: joel
'''

import unittest
import numpy as np
import scipy.special.orthogonal as sso 
import pypwdg.utils.quadrature as puq
import pypwdg.utils.polynomial as pup

class TestJacobi(unittest.TestCase):
    def testOrthogonal(self):
        N = 4
        MaxAB = 4
        x,w = puq.legendrequadrature(N + MaxAB)
        x = x.ravel()
        n = np.arange(N, dtype = float)
        for a in range(MaxAB):
            for b in range(MaxAB):
                # first lets test some orthogonality:
                P = pup.jacobidnorm(N-1, a, b, 0, x) # get the first N jacobi polynomials evaluated at x                
                jw = w * x**b * (1-x)**a # combine quadrature weight with orthogonality weight
                PP = np.dot(P.T * jw, P)
#                norm2 = (1 /(2*n+a+b+1)) * (sso.poch(n + 1, a) / sso.poch(n + b + 1, a))
#                np.testing.assert_almost_equal(PP, np.diag(norm2))
                np.testing.assert_almost_equal(PP, np.eye(N,N))
    
    def testDerivative(self):
        N = 4
        MaxAB = 4
        t = np.linspace(0,1,20)
        h = 1E-8
        for a in range(MaxAB):
            for b in range(MaxAB):                
                # now lets test the derivatives:
                PD = pup.jacobid(N-1, a, b, 1, t)
                PDh = (pup.jacobid(N-1, a, b, 0, t+h) - pup.jacobid(N-1, a, b, 0, t-h)) / (2*h)
                np.testing.assert_almost_equal(PD, PDh, decimal=4)
                            

class TestDubiner(unittest.TestCase):
    
    def testValues(self):
        N = 5        
        for k in range(N):
            x,w = puq.trianglequadrature(k+1)
            dt = pup.DubinerTriangle(k, x)
            V = dt.values()
            n = ((k+1)*(k+2))/2
            self.assertEquals(V.shape[1], n) # test that we get the correct number of basis functions
            VV = np.dot(V.T, w * V)
            np.testing.assert_almost_equal(VV, np.eye(n)) # test that they are orthonormal
            
    def testDerivs(self):
        N = 5
        h = 1E-8
        for k in range(0,N):
            x,w = puq.trianglequadrature(k+1)
            dt = pup.DubinerTriangle(k,x)
            dtV = dt.values()
            dtxh = pup.DubinerTriangle(k, x+[h,0])
            dtxhV = dtxh.values()
            dtyh = pup.DubinerTriangle(k, x+[0,h])
            dtyhV = dtyh.values()
            dtD = dt.derivs()
            dtxhD = (dtxhV - dtV)/h
            dtyhD = (dtyhV - dtV)/h
            
            np.testing.assert_almost_equal(dtD[0],dtxhD, decimal=4)
            np.testing.assert_almost_equal(dtD[1],dtyhD, decimal=4)

    def testOrthonormal(self):
        ''' Test that the Dubiner polynomials are L^2-orthonormal over the reference triangle'''
        N = 8
        for k in range(0, N):
            x,w = puq.trianglequadrature(k+1)
            dt = pup.DubinerTriangle(k,x)
            dtV = dt.values()
            l2 = np.dot(dtV.T, w * dtV)
            np.testing.assert_almost_equal(l2, np.eye(dtV.shape[1],dtV.shape[1]))

class TestLaplacian(unittest.TestCase):
    
    def testLaplaceZero(self):
        ''' Test that the linear polynomials will have a zero laplacian'''
        L = pup.laplacian(1)
        np.testing.assert_almost_equal(L, 0, decimal=5) 

    def testLaplacian(self):
        ''' Use finite differences to check the Lapalcian calculations'''
        N = 6
        h = 1E-4
        x,w = puq.trianglequadrature(5)
        for k in range(1,N):
            v = pup.DubinerTriangle(k, x).values()
            L = pup.laplacian(k)
            Lv = np.dot(v, L)
            Lhv = (sum([pup.DubinerTriangle(k, x+xh).values() for xh in ([0,h],[0,-h],[h,0],[-h,0])]) - 4*v)/(h**2)
            np.testing.assert_almost_equal(Lv, Lhv, decimal=3)