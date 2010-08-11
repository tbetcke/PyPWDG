'''
Created on Aug 11, 2010

@author: joel
'''
from pypwdg.mesh.meshutils import MeshQuadratures
from pypwdg.core.vandermonde import LocalVandermondes
from pypwdg.core.bases import PlaneWaves
from pypwdg.mesh.structure import StructureMatrices
from pypwdg.utils.timing import print_timing
from pypwdg.core.assembly import Assembly

import numpy


@print_timing
def impedanceSystem(mesh, k, g, localquads, elttobasis, usecache=True, alpha = 1.0/2, beta = 1.0/2, delta = 1.0/2):
    """ Assemble the stiffness and load matrices for the PW DG method with UWVF parameters
    
        k: wave number
        g: boundary data (should have a values method and a derivs method - see .core.bases.PlaneWaves)
        localquads: local quadrature rule for each face
        elttobasis: list of bases for each element
    """
    print "Mesh has %s elements"%mesh.nelements
    print "k = %s"%k
    print "%s basis functions"%sum([b.n for bs in elttobasis for b in bs ])
    print "%s quadrature points"%len(localquads[1])
    
    boundaryentities = []
    SM = StructureMatrices(mesh, boundaryentities)
    jk = 1j * k
    jki = 1/jk
    mqs = MeshQuadratures(mesh, localquads)
    lv = LocalVandermondes(mesh, elttobasis, mqs.quadpoints, usecache)
    stiffassembly = Assembly(lv, lv, mqs.quadweights)        
    SI = stiffassembly.assemble(numpy.array([[jk * alpha * SM.JD,   -SM.AN], 
                                             [SM.AD,                -beta*jki * SM.JN]]))
    
    # now for the boundary contribution
    #impedance boundary conditions
    SB = stiffassembly.assemble(numpy.array([[jk * (1-delta) * SM.B, -delta * SM.B],
                                             [(1-delta) * SM.B,      -delta * jki * SM.B]]))
        

    gelts = [[g]] * mesh.nelements
    gv = LocalVandermondes(mesh, gelts, mqs.quadpoints)
    
    loadassembly = Assembly(lv, gv, mqs.quadweights)
    # todo - check the cross terms.  Works okay with delta = 1/2.  
    GB = loadassembly.assemble(numpy.array([[jk * (1-delta) * SM.B,  (1-delta) * SM.B], 
                                            [-delta * SM.B,          -delta * jki * SM.B]]))
    
    S = SM.sumfaces(SI + SB)     
    G = SM.sumrhs(GB)
    return S,G    
