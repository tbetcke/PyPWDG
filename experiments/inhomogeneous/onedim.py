'''
Created on Jul 11, 2011

@author: joel
'''
#import instant.config
#instant.config.get_status_output('env')
#print instant.config.get_status_output('echo $PATH')
#
##instant.config.check_and_set_swig_binary("swig", "/opt/local/bin")
#print instant.config.get_swig_version()
#print instant.config.get_swig_binary()
#import sys
#print sys.path

import pypwdg.setup.computation as psc

import scipy.sparse as ss
import numpy as np
import matplotlib.pyplot as mp

from dolfin import *
import math

def ssmatrix(A):
    Ap, Ai, Ad = A.data()
    return ss.csr_matrix((Ad, Ai, Ap))

def fenics1dhelmholtz(N, p, k):
#N = 100
#p = 5
#k = 30
    mesh = UnitInterval(N)
    V = FunctionSpace(mesh, 'Lagrange', p)
    u = TrialFunction(V)
    v = TestFunction(V)
    gr = Expression("cos(k*x[0])")
    gi = Expression("sin(k*x[0])")
    nx = Expression("exp(2*x[0]*(1-x[0])+1)")
#    nx = Constant(1.0)
    n = FacetNormal(mesh)
    
    A = as_matrix(((gr, gi),(-gi,gr)))
    
    gr.k = k
    gi.k = k
    
    a = (dot(grad(u), grad(v)) - k**2 * nx**2 * (inner(u,v)))*dx
    b = k * dot(u,v) * ds 
    
    f = (gr * v * (1.0+n))*ds
    g = (gi * v * (1.0+n))*ds
    
    A = assemble(a)
    B = assemble(b)
    F = assemble(f)
    G = assemble(g)
    
    M = ssmatrix(A) + 1j * ssmatrix(B)
    L = 1j * k * (F.data() + 1j * G.data())
    X = psc.DirectSolver().bestSolve(M, L)
    xr = Function(V)
    xi = Function(V)
    xr.vector()[:] = X.real.copy()
    xi.vector()[:] = X.imag.copy()
    return xr, xi

def errornorm(ue, uh, Ve = None):
    if Ve is None:
        Ve = ue.function_space()
    else:
        ue = interpolate(ue, Ve)
    uh = interpolate(uh, Ve)
    e = Function(Ve)
    # Subtract degrees of freedom for the error field
    e.vector()[:] = ue.vector().array() - \
                       uh.vector().array()
    error = e**2*dx
    return sqrt(assemble(error, mesh=Ve.mesh()))

def standardconvergence(kk):
    pexact = 6
    ur, ui = fenics1dhelmholtz(1024,pexact,kk)
    
    for pp in range(1,6):
        for NN in 2**np.arange(1,10):
            uhr, uhi = fenics1dhelmholtz(NN,pp,kk)
            print pp,NN, len(uhr.vector()), sqrt(errornorm(ur,uhr)**2 + errornorm(ui, uhi)**2)

def plotfn(f):
    t = np.linspace(0,1,200)
    y = np.array([f(x) for x in t])
    mp.plot(t,y)
    return t
    
def polyfit(f, deg):
    N = (deg+1) * 10
#    N = 100
    t = np.linspace(0,1, N)
    y = np.array([f(x) for x in t])
    z, residuals, rank, singular_values, rcond = np.polyfit(t, y, deg, full=True)
#    print residuals, rank, singular_values, rcond
    err = math.sqrt(residuals / N)
    return np.poly1d(z), err


    
kk = 4
pexact = 6
ur, ui = fenics1dhelmholtz(1024,pexact,kk)
u = lambda x: ur(x) + 1j * ui(x)
t = plotfn(u)
for p in range(7,18,2):
    up, e = polyfit(u, p)
    print p, e
    mp.plot(t, up(t), '--')


#mesh1 = UnitInterval(1)
#p = 10
#V = FunctionSpace(mesh1, 'Lagrange', p)
#upr = project(ur, V)
#uir = interpolate(ur,V)
#print errornorm(ur, upr)
#print errornorm(ur, uir)
#plotfn(upr)
#plotfn(uir)

mp.show()



#plot(ur)

#
##
#finemesh = UnitInterval(N*p)
#Vfine = FunctionSpace(finemesh, 'Lagrange', 1)
#xfine = interpolate(xfn, Vfine)
#plot(xfine)
##    
##    
#
#if __name__ == '__main__':
#    fenics1dhelmholtz(20,1,5)