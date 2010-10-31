'''
Created on Aug 11, 2010

@author: joel
'''
from pypwdg.mesh.meshutils import MeshQuadratures
from pypwdg.core.vandermonde import LocalVandermondes
from pypwdg.utils.timing import print_timing
from pypwdg.core.assembly import Assembly
import pypwdg.mesh.structure as pms

from pypwdg.parallel.decorate import parallel, tuplesum

import numpy

@parallel(None, reduceop=tuplesum)
@print_timing
def assemble(mesh, k, quadrule, elttobasis, bnddata, params):
        
    stiffassembly,loadassemblies,vandermondes, bndvs=init_assembly(mesh,quadrule,elttobasis,bnddata,usecache=True)
    
    S=assemble_int_faces(mesh, k, stiffassembly, params)
    f=0
    
    for (id, bdycondition), loadassembly in zip(bnddata.items(), loadassemblies):
        (Sb,fb)=assemble_bnd(mesh, k, id, bdycondition, stiffassembly, loadassembly, params)
        S=S+Sb
        f=f+fb
    return S, f, vandermondes, bndvs

def init_assembly(mesh,localquads,elttobasis,bnddata,usecache=True):

    mqs = MeshQuadratures(mesh, localquads)
    lv = LocalVandermondes(mesh, elttobasis, mqs.quadpoints, usecache)
    stiffassembly = Assembly(lv, lv, mqs.quadweights) 
    
    loadassemblies = []
    bndvs=[]
    for data in bnddata.values():
        bndv = LocalVandermondes(mesh, [[data]] * mesh.nelements, mqs.quadpoints)        
        loadassemblies.append(Assembly(lv, bndv, mqs.quadweights))
        bndvs.append(bndv)

    return (stiffassembly,loadassemblies,lv,bndvs)

def assemble_int_faces(mesh, k, stiffassembly, params):
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
    AJ = pms.AveragesAndJumps(mesh)    
    SI = stiffassembly.assemble(numpy.array([[jk * alpha * AJ.JD,   -AJ.AN], 
                                             [AJ.AD,                -beta*jki * AJ.JN]]))
    
    
    return pms.sumfaces(mesh,SI)

def assemble_bnd(mesh, k, id, bnd_condition, stiffassembly, loadassembly, params):
    
    #mqs = MeshQuadratures(mesh, localquads)
    #lv = LocalVandermondes(mesh, elttobasis, mqs.quadpoints, usecache)
           

    
    delta=params['delta']

    l_coeffs=bnd_condition.l_coeffs
    r_coeffs=bnd_condition.r_coeffs
    
    B = mesh.entityfaces[id]
        
    SB = stiffassembly.assemble(numpy.array([[l_coeffs[0]*(1-delta) * B, (-1+(1-delta)*l_coeffs[1])*B],
                                             [(1-delta*l_coeffs[0]) * B,      -delta * l_coeffs[1]*B]]))
        

    # todo - check the cross terms.  Works okay with delta = 1/2.  
    GB = loadassembly.assemble(numpy.array([[(1-delta) *r_coeffs[0]* B,  (1-delta) * r_coeffs[1]*B], 
                                            [-delta*r_coeffs[0]* B,          -delta * r_coeffs[1]*B]]))
        
    S = pms.sumfaces(mesh,SB)     
    G = pms.sumrhs(mesh,GB)
    return S,G    

@parallel(None, reduceop=tuplesum)
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
    
    jk = 1j * k
    jki = 1/jk
    mqs = MeshQuadratures(mesh, localquads)
    lv = LocalVandermondes(mesh, elttobasis, mqs.quadpoints, usecache)
    AJ = pms.AveragesAndJumps(mesh)
    stiffassembly = Assembly(lv, lv, mqs.quadweights)        
    SI = stiffassembly.assemble(numpy.array([[jk * alpha * AJ.JD,   -AJ.AN], 
                                             [AJ.AD,                -beta*jki * AJ.JN]]))
#    print "SI13,75 ",SI.tocsr()[13,75]
    
    # now for the boundary contribution
    #impedance boundary conditions
    B = mesh.boundary
    SB = stiffassembly.assemble(numpy.array([[jk * (1-delta) * B, -delta * B],
                                             [(1-delta) * B,      -delta * jki * B]]))
        
    print "Cached vandermondes %s"%lv.getCachesize()
    
    gelts = [[g]] * mesh.nelements
    gv = LocalVandermondes(mesh, gelts, mqs.quadpoints)
    
    loadassembly = Assembly(lv, gv, mqs.quadweights)
    # todo - check the cross terms.  Works okay with delta = 1/2.  
    GB = loadassembly.assemble(numpy.array([[jk * (1-delta) * B,  (1-delta) * B], 
                                            [-delta * B,          -delta * jki * B]]))
        
    S = pms.sumfaces(mesh,SI + SB)     
    G = pms.sumrhs(mesh,GB)
#    print "S 4,25",S.tocsr()[4,25]    
    return S,G    
