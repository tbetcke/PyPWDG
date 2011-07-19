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
import numpy as n

from dolfin import *

def ssmatrix(A):
    Ap, Ai, Ad = A.data()
    return ss.csr_matrix((Ad, Ai, Ap))

#def fenics1dhelmholtz(N, p, k):
N = 100
p = 5
k = 30
mesh = UnitInterval(N)
V = FunctionSpace(mesh, 'Lagrange', p)
u = TrialFunction(V)
v = TestFunction(V)
gr = Expression("cos(k*x[0])")
gi = Expression("sin(k*x[0])")
nx = Expression("exp(4*x[0]*(1-x[0])+1)")

n = FacetNormal(mesh)

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
print X
x = uBLASVector(len(X))
x[:] = X.real.copy()
xfn = Function(V,x) 
plot(xfn)
#
finemesh = UnitInterval(N*p)
Vfine = FunctionSpace(finemesh, 'Lagrange', 1)
xfine = interpolate(xfn, Vfine)
plot(xfine)
#    
#    
#
#if __name__ == '__main__':
#    fenics1dhelmholtz(20,1,5)