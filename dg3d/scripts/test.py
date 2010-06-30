'''
Created on May 6, 2010

@author: joel
'''
import sys
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


if __name__ == '__main__':
    dir = na([[1,0,0]])
    k = 20
    maxvol = 0.04
    print k, maxvol
    g = dg3d.impedance(dir,k)
    gg = dg3d.planeWaves(dir,k)[0]
    s = dg3d.solver(dg3d.cubemesh(maxvol), 3, 10, k).solve(g)
    v = dg3d.Visualiser(s,gg)
    v.showuaveragereal()
    sys.exit(0)
    
    