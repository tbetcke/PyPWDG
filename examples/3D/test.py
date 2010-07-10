'''
Created on May 6, 2010

@author: joel
'''
import sys
sys.path.append('../..')

import numpy
from numpy import array as na
import PyPWDG.dg3d as dg3d
import PyPWDG.dg3d.visualisation

def basiccube():
    dir = na([[1,0,0]])
    k = 15
    maxvol = 0.02
    g = dg3d.impedance(dir,k)
    gg = dg3d.planeWaves(dir,k)[0]
    s = dg3d.solver(dg3d.cubemesh(maxvol), 2, 12, k).solve(g)
    v = dg3d.Visualiser(s,gg)

def numpysum():
    import time
    n = 1000
    A = numpy.mat(numpy.reshape(range(n*n), (n,n)), dtype=numpy.float64)
    t = []
    t.append(time.time())
    B1 = A + A +A + A
    t.append(time.time())
    A2 = [A,A,A,A]
    t.append(time.time())
    B2 = numpy.sum(A2, axis=0)
    t.append(time.time())
    l = len(A2)
    B3 = numpy.reshape(numpy.dot(numpy.ones(l), numpy.reshape(A2, (l, -1))), (n,n))
    t.append(time.time())
    op = lambda a,b:a+b
    B4 = reduce(op, A2)
    t.append(time.time())
     
    print na(t[1:]) - na(t[:-1])

if __name__ == '__main__':
    print "hello"
    dir = na([[1,0,0]])
    k = 20
    maxvol = 0.04
    print k, maxvol
    g = dg3d.impedance(dir,k)
    gg = dg3d.planeWaves(dir,k)[0]
    s = dg3d.solver(dg3d.cubemesh(maxvol), 3, 10, k)
    
#    s.solve(g)
#    
#    v = dg3d.Visualiser(s,gg)
#    v.showuaveragereal()
#    sys.exit(0)
    
    
