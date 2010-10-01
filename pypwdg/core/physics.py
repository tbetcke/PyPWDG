'''
Created on Aug 11, 2010

@author: joel
'''
from pypwdg.mesh.meshutils import MeshQuadratures
from pypwdg.core.vandermonde import LocalVandermondes
from pypwdg.utils.timing import print_timing
from pypwdg.core.assembly import Assembly
from pypwdg.core.bases import EmptyBasis

from pypwdg.parallel.decorate import parallel, tuplesum

import numpy

# is this ugly.  However, the right way to do it is to distribute the SM object ... coming soon.
def impsysscatter(n):
    def splitargs(mesh, SM,k, g, localquads, elttobasis, usecache=True, alpha = 1.0/2, beta = 1.0/2, delta = 1.0/2):
        mesh.partition(n)    
        return [((mesh, SM.withFP(facepart), k, g, localquads, elttobasis, usecache, alpha, beta, delta),{}) for facepart in mesh.facepartitions]
    return splitargs

def init_assembly(mesh,localquads,elttobasis,bnddata,usecache=True):

    mqs = MeshQuadratures(mesh, localquads)
    lv = LocalVandermondes(mesh, elttobasis, mqs.quadpoints, usecache)
    stiffassembly = Assembly(lv, lv, mqs.quadweights) 
    
    ftoe=lambda f: mesh.faces[f][0] # Mapping from a faceId f to the corresponding elementId
    gelts=[ [EmptyBasis(1)] for _ in range(mesh.nelements) ]
    for bndface in mesh.bndfaces: gelts[ftoe(bndface)]=[bnddata[mesh.bnd_entities[bndface]]]
    #gelts=[[bnddata[29]]]*mesh.nelements
    bndv = LocalVandermondes(mesh, gelts, mqs.quadpoints)
    
    loadassembly = Assembly(lv, bndv, mqs.quadweights)

    return (stiffassembly,loadassembly)

def assemble_int_faces(mesh, SM, k, stiffassembly, params):
    "Assemble the stiffness matrix for the interior faces"

    print "Mesh has %s elements"%mesh.nelements
    print "k = %s"%k
    #print "%s basis functions"%sum([b.n for bs in elttobasis for b in bs ])
    #print "%s quadrature points"%len(localquads[1])
    
    alpha=params['alpha']
    beta=params['beta']
    
    jk = 1j * k
    jki = 1/jk
    #mqs = MeshQuadratures(mesh, localquads)
    #lv = LocalVandermondes(mesh, elttobasis, mqs.quadpoints, usecache)      
    SI = stiffassembly.assemble(numpy.array([[jk * alpha * SM.JD,   -SM.AN], 
                                             [SM.AD,                -beta*jki * SM.JN]]))
    
    return SM.sumfaces(SI)

def assemble_bnd(mesh, SM, k, bnd_conditions, id, stiffassembly, loadassembly, params):
    
    #mqs = MeshQuadratures(mesh, localquads)
    #lv = LocalVandermondes(mesh, elttobasis, mqs.quadpoints, usecache)
           

    
    delta=params['delta']

    l_coeffs=bnd_conditions[id].l_coeffs
    r_coeffs=bnd_conditions[id].r_coeffs
    

    
    SB = stiffassembly.assemble(numpy.array([[l_coeffs[0]*(1-delta) * SM.BE[id], (-1+(1-delta)*l_coeffs[1])*SM.BE[id]],
                                             [(1-delta*l_coeffs[0]) * SM.BE[id],      -delta * l_coeffs[1]*SM.BE[id]]]))
        

    # todo - check the cross terms.  Works okay with delta = 1/2.  
    GB = loadassembly.assemble(numpy.array([[-(1-delta) *r_coeffs[0]* SM.BE[id],  -(1-delta) * r_coeffs[1]*SM.BE[id]], 
                                            [delta*r_coeffs[0]* SM.BE[id],          delta * r_coeffs[1]*SM.BE[id]]]))
        
    S = SM.sumfaces(SB)     
    G = SM.sumrhs(GB)
    return S,G    


@parallel(impsysscatter, reduceop=tuplesum)
def impedanceSystem(mesh, SM, k, g, localquads, elttobasis, usecache=True, alpha = 1.0/2, beta = 1.0/2, delta = 1.0/2):
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
        
    print "Cached vandermondes %s"%lv.getCachesize()
    
    gelts = [[g]] * mesh.nelements
    gv = LocalVandermondes(mesh, gelts, mqs.quadpoints)
    
    loadassembly = Assembly(lv, gv, mqs.quadweights)
    # todo - check the cross terms.  Works okay with delta = 1/2.  
    GB = loadassembly.assemble(numpy.array([[jk * (1-delta) * SM.B,  (1-delta) * SM.B], 
                                            [-delta * SM.B,          -delta * jki * SM.B]]))
        
    S = SM.sumfaces(SI + SB)     
    G = SM.sumrhs(GB)
        
    return S,G    
