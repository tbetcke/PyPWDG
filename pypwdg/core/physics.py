'''
Created on Aug 11, 2010

@author: joel
'''
from pypwdg.mesh.meshutils import MeshQuadratures
from pypwdg.core.vandermonde import LocalVandermondes, LocalInnerProducts, ElementVandermondes
from pypwdg.utils.timing import print_timing
from pypwdg.core.assembly import Assembly
import pypwdg.mesh.structure as pms
import pypwdg.utils.sparse as pus
import scipy.sparse as ss
import numpy as np

from pypwdg.parallel.decorate import parallel, tuplesum

import numpy

@parallel(None, reduceop=tuplesum)
@print_timing
def assemble(mesh, k, lv, bndvs, mqs, elttobasis, bnddata, params, emqs, dovols):
        
    stiffassembly,loadassemblies = init_assembly(mesh,lv, bndvs, mqs,elttobasis,bnddata)
    
    
    S=assemble_int_faces(mesh, k, stiffassembly, params)
    f=0
    
    if dovols:
        V = assemble_volume_terms(mesh, k, elttobasis, emqs, stiffassembly)
        S+=V
    
    for (id, bdycondition), loadassembly in zip(bnddata.items(), loadassemblies):
        (Sb,fb)=assemble_bnd(mesh, k, id, bdycondition, stiffassembly, loadassembly, params)
        S=S+Sb
        f=f+fb
    return S, f

def init_assembly(mesh,lv,bndvs, mqs,elttobasis,bnddata):

    stiffassembly = Assembly(lv, lv, mqs.quadweights) 
    
    loadassemblies = []
    for bndv in bndvs:
        loadassemblies.append(Assembly(lv, bndv, mqs.quadweights))

    return (stiffassembly,loadassemblies)

def assemble_int_faces(mesh, k, stiffassembly, params):
    "Assemble the stiffness matrix for the interior faces"

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


def assemble_volume_terms(mesh, k, elttobasis, emqs, stiffassembly):
    ev = ElementVandermondes(mesh, elttobasis, emqs)
    L2 = LocalInnerProducts(ev.getValues, ev.getValues, emqs.quadweights)
    H1 = LocalInnerProducts(ev.getDerivs, ev.getDerivs, emqs.quadweights, ((0,2),(0,2)))
    d = np.zeros(mesh.nelements)
    d[mesh.partition] = 1
    csrelts = ss.dia_matrix((d, [0]), shape = (mesh.nelements,)*2).tocsr()
    L2P = pus.createvbsr(csrelts, L2.product, elttobasis.getSizes(), elttobasis.getSizes())
    H1P = pus.createvbsr(csrelts, H1.product, elttobasis.getSizes(), elttobasis.getSizes())
    
    Zero = ss.csr_matrix((mesh.nfaces, mesh.nfaces))
    B = pms.sumfaces(mesh, stiffassembly.assemble([[Zero,Zero],[mesh.facepartition,Zero]]))
    
    return H1P - k**2 * L2P - B
      

