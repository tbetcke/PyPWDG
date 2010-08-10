'''
Created on Aug 9, 2010

@author: joel
'''

import numpy

from PyPWDG.mesh.meshutils import MeshQuadratures
from PyPWDG.core.vandermonde import LocalVandermondes, LocalInnerProducts
from PyPWDG.core.bases import PlaneWaves
from PyPWDG.mesh.structure import StructureMatrices
from PyPWDG.utils.sparse import createvbsr
from PyPWDG.utils.timing import print_timing

class Assembly(object):
    def __init__(self, lv, rv, mqs):
        DD = LocalInnerProducts(lv.getValues, rv.getValues, mqs)
        DN = LocalInnerProducts(lv.getValues, rv.getDerivs, mqs)
        ND = LocalInnerProducts(lv.getDerivs, rv.getValues, mqs)
        NN = LocalInnerProducts(lv.getDerivs, rv.getDerivs, mqs)
        
        self.ips = numpy.array([[DD,DN],[ND,NN]])
        self.numleft = lv.numbases
        self.numright = rv.numbases
        
    def assemble(self, structures):
        return sum([createvbsr(structures[i,j],self.ips[i,j].product, self.numleft, self.numright) for i in [0,1] for j in [0,1]])
            

def impedanceSystem(mesh, k, g, localquads, dirs, alpha = 1.0/2, beta = 1.0/2, delta = 1.0/2):
    boundaryentities = []
    SM = StructureMatrices(mesh, boundaryentities)
    jk = 1j * k
    jki = 1/jk
    mqs = MeshQuadratures(mesh, localquads)
    pws = PlaneWaves(dirs, k)
    elttobasis = [[pws]] * mesh.nelements
    lv = LocalVandermondes(mesh, elttobasis, mqs.quadpoints)
    stiffassembly = Assembly(lv, lv, mqs.quadweights)        
    SI = stiffassembly.assemble(numpy.array([[jk * alpha * SM.JD,   -SM.AN], 
                                             [SM.AD,                -beta*jki * SM.JN]]))
    
    SB = stiffassembly.assemble(numpy.array([[jk * (1-delta) * SM.boundary, -delta * SM.boundary],
                                             [(1-delta) * SM.boundary,      -delta * jki * SM.boundary]]))
        
    # now for the boundary contribution
    #impedance boundary conditions
    print SI, SB
    S = SM.sumfaces(SI + SB)     

    # lets reuse what we have:
    gelts = [[g]] * mesh.nelements
    gv = LocalVandermondes(mesh, gelts, mqs.quadpoints)
    
    loadassembly = Assembly(lv, gv, mqs.quadweights)
    # todo - check the cross terms.  Works okay with delta = 1/2.  
    GB = loadassembly.assemble(numpy.array([[jk * (1-delta) * SM.boundary,  (1-delta) * SM.boundary], 
                                            [-delta * SM.boundary,          -delta * jki * SM.boundary]]))
    
    G = SM.sumrhs(GB)
    return S,G    


