'''
Created on Jul 11, 2011

@author: joel
'''
import instant.config
instant.config.get_status_output('env')
print instant.config.get_status_output('echo $PATH')

#instant.config.check_and_set_swig_binary("swig", "/opt/local/bin")
print instant.config.get_swig_version()
print instant.config.get_swig_binary()
import sys
print sys.path

from dolfin import *

def fenics1dhelmholtz(N, p, k):
    mesh = UnitInterval(N)
    V = FunctionSpace(mesh, 'Lagrange', p)
    VC = V * V
    ur,ui = TrialFunction(VC)
    vr,vi = TestFunction(VC)
    gr = Expression("k*cos(k*x[0])")
    gi = Expression("k*sin(k*x[0])")
    n = FacetNormal(mesh)
    
    gr.k = k
    gi.k = k
    
    a = (dot(grad(ur), grad(vr)) + dot(grad(ui), grad(vi)) \
         - k**2 * (inner(ur,vr) + inner(ui,vi)))*dx\
         + k * (dot(ur,vi) + dot(ui,vr)) * ds 
    
    L = (gi * vr * (1.0+n) + gr *vi*(1.0+n))*ds
    
    problem = VariationalProblem(a,L)
    (Ur,Ui) = problem.solve().split()
    plot(Ur)
    plot(Ui)
    interactive()

    
    
    

if __name__ == '__main__':
    fenics1dhelmholtz(50,1,10)