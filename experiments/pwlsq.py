'''
Created on Sep 9, 2010

@author: joel
'''

import pypwdg.core.bases as pcb
import numpy as np
import pypwdg.utils.quadrature as puq
import scipy.optimize as so
import math
import numpy.linalg as sl
import scipy.special as ss
import matplotlib.pyplot as mp

def thetatodirs(thetas):
    thetas = thetas.reshape(-1,1)
    return np.hstack((np.cos(thetas), np.sin(thetas)))


def planewavefn(points, k, params):
    """ Evaluate a function composed of N plane waves
     
        params: a N x 2 array.  Column 1 is the coefficients, column 2 is the thetas 
        k: wave number
        points: where to evaluate the planewaves
        
        returns a len(points) array of values
    """  
    params = params.reshape(-1, 2)
    return np.dot(pcb.PlaneWaves(thetatodirs(params[:,1]), k).values(points), params[:,0]) 

def l2norm(v, w):
    """ Helper function to calculate the l2 norm"""
    return math.sqrt(np.sum(v.conj() * v * w))

def rectquad(rect, nq):
    c1, c2 = rect
    p,w = puq.squarequadrature(nq)
    rp = p * (c2 - c1) + c1
    rw = (w * np.prod(c2 - c1))
    return rp,rw

def bestfit(u, k, rect, numpw, nq):
    """ Attempt to find a best-fit function composed of numpw planewaves
    
        u: function to fit
        k: wave number
        rect: corners of rectangle to calculate L^2 norm over
        nq: number of quadrature points
    """
    rp,rw = rectquad(rect, nq)
    urp = u(rp).flatten()
    fn = lambda params: l2norm(urp - planewavefn(rp, k, params), rw) # This is the objective function
    
    initialtheta = np.arange(numpw).reshape((-1,1)) * 2*math.pi / numpw
    initialguess = np.hstack((np.zeros((numpw,1)), initialtheta)) # Take initial coefficients as 1s 
    
    print fn(initialguess)
    
    xopt= so.fmin_powell(fn, initialguess).reshape(-1,2)
    print "Coefficients: ", xopt[:,0]
    print "Thetas: ", xopt[:,1]
    print "Norm: ", fn(xopt)
    
def bestfit2(u, k, rect, numpw, nq, printresults = True):
    """ A potentially better best fit algorithm
    
        Just does the non-linear optimisation over the directions.  Uses least squares to work out the coefficients
        
        u: function to fit
        k: wave number
        rect: corners of rectangle to calculate L^2 norm over
        numpw: number of plane waves to use
        nq: number of quadrature points
    """
    rp,rw = rectquad(rect, nq)
    urp = u(rp).flatten() * np.sqrt(rw)
    def lstsqerr(thetas, retcoeffs = False):
        PW = pcb.PlaneWaves(thetatodirs(thetas), k)
        PWp = PW.values(rp) * np.sqrt(rw.reshape(-1,1))
        c = sl.lstsq(PWp, urp)[0]
        err = np.dot(PWp, c) - urp
        l2err = math.sqrt(np.sum(err.conj() * err))
        if retcoeffs: return l2err, c
        return l2err 
        
    initialtheta = np.arange(numpw) * 2*math.pi / numpw
    thetaopt = so.fmin_powell(lstsqerr, initialtheta, disp=False)
    l2err, c = lstsqerr(thetaopt, True)
    if printresults:
        print "Optimal thetas: ", thetaopt
        print "Optimal coeffs: ", c
        print "L2 error: ", l2err    
    return l2err, thetaopt, c

def fittopw():
    """ Example of attemping to fit a plane wave """
    k = 10
    rect = (np.array([1,1]), np.array([2,2]))
    u = pcb.PlaneWaves(np.array([[0.0,1.0]]), k).values
    bestfit(u,k,rect, 2, 20)
    bestfit2(u,k,rect,2,20)

def fourierbessel(k,n,points):
    r = np.sqrt(np.sum(points**2, axis=1))
    theta = np.arctan(points[:,1] / points[:,0])
    return ss.jn(n, k*r)*np.exp(1j * n * theta)

def fitfb():
    MAXPW = 15
    k = 10
    n = 5
    rect = (np.array([1,1]), np.array([2,2]))
    u = lambda points: fourierbessel(k,n,points)
    npws = []
    l2errs = []
    for npw in range(1,MAXPW): 
        l2err, thetaopt, c = bestfit2(u,k,rect,npw,15, False)
        print npw, l2err
        npws.append(npw)
        l2errs.append(l2err)
    
    mp.semilogy(npws, l2errs)
    mp.show()
        
    
    
if __name__ == '__main__':
#    fittopw()
    fitfb()