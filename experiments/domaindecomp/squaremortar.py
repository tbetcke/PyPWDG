import pypwdg.core.bases as pcb
import pypwdg.mesh.mesh as pmm
import pypwdg.core.boundary_data as pcbd
import pypwdg.core.bases.utilities as pcbu
import pypwdg.setup.problem as psp
import pypwdg.setup.computation as psc
import pypwdg.setup.indirect as psi
import pypwdg.core.physics as pcp
import pypwdg.output.solution as pos
import pypwdg.test.utils.mesh as tum
import numpy as np
np.set_printoptions(threshold='nan')

import pypwdg.parallel.main

from numpy import array,sqrt

@ppd.distribute()
class SkeletonPartition(pmm.Partition):
    def __init__(self, meshview, skelemeshinfo, skelemeshtopology, skelefaceidx):
        pmm.Partition.__init__(self, skelemeshinfo, skelemeshtopology, meshview.facepartition * skelefaceidx)
        

def skeletonMesh(mesh, skeletontag):
    ''' Creates a codimension 1 submesh consisting of the faces associated with the boundary highlighted by skeletontag
        Also returns the element->face map from the submesh back to the original mesh
    '''
    
    skeletonfaceidx = mesh.topology.facentities == skeletontag  
    skeletonfaces = mesh.faces[skeletonfaceidx]
    meshinfo = pmm.SimplicialMeshInfo(mesh.nodes, skeletonfaces, None, {}, mesh.dim -1)
    topology = pmm.Topology(meshinfo)
    return pmm.MeshView(meshinfo, topology), skelefacemtx * np.arange(mesh.nfaces)

def skeletonBasis(skelemesh, basisrule):
    etob = {}
    ei = pcbu.ElementInfo(mesh, 0)
    for e in skelemesh.partition:
        etob[e] = basisrule.populate(ei.info(e))
    return pcbu.CellToBases(etob)
    

class SkeletonFaceToBasis(object):
    def __init__(self, mesh, skeleelttobasis):
        self.mesh = mesh
        self.elttobasis = elttobasis
        self.numbases = elttobasis.getSizes()[mesh.ftoe]
        self.indices = elttobasis.getIndices()[mesh.ftoe]
    
    def evaluate(self, faceid, points):
        e = self.mesh.ftoe[faceid] 
        normal = self.mesh.normals[faceid]
        vals = self.elttobasis.getValues(e, points)
        derivs = self.elttobasis.getDerivs(e, points, normal)
        return (vals,derivs)

nparts = 3
k = 10
n = 5
g = pcb.FourierHankel([-1,-1], [0], k)
bdytag = "BDY"
bnddata={bdytag:pcbd.dirichlet(g)}

bounds=array([[0,1],[0,1]],dtype='d')
npoints=array([200,200])
meshinfo = tum.regularsquaremeshinfo(n, bdytag)
topology = pmm.Topology(meshinfo)

parts = [pmm.Partition(meshinfo, topology, partition, i) for i,partition in enumerate(meshinfo.partition(nparts))]

cutfaces = (part.cutfaces.nonzero()[0] for part in parts)
cutvertices = np.concatenate([meshinfo.faces[fs] for fs in cutfaces])

boundaries = list(meshinfo.boundaries)
boundaries.append(('INTERNAL', cutvertices))

meshinfo2 = pmm.SimplicialMeshInfo(meshinfo.nodes, meshinfo.elements, meshinfo.elemIdentity, boundaries, meshinfo.dim)
topology2 = pmm.Topology(meshinfo)
meshes = [pmm.MeshView(meshinfo2, topology2, part) for part in parts]

skeletonmeshinfos = [skeletonMesh(mesh, 'INTERNAL') for mesh in meshes]

skeletob = skeletonBasis(skelemesh)
    
problem = psp.Problem(mesh, k, bnddata)
problem.bdyinfo['INTERNAL'] = (pcbd.BoundaryCondition([-1j*k, 1], [-1j*k, 1]), boundaryftob)



computation = psc.Computation(problem, pcb.planeWaveBases(2,k,5), pcp.HelmholtzSystem, 13)

solbrutal = computation.solution(psi.BrutalSolver(np.complex, psi.DomainDecompOperator(mesh)).solve)
#print "DD solve"
#soldd = computation.solution(psi.IndirectSolver(np.complex, psi.DomainDecompOperator(mesh)).solve)
#print "Block precond solve"
#solindirect = computation.solution(psi.IndirectSolver(np.complex, psi.BlockPrecondOperator(mesh)).solve)
print "Direct solve"
soldirect = computation.solution(psc.DirectSolver().solve)

#print soldirect.x
#print solbrutal.x
#print np.abs(soldirect.x - solbrutal.x)
print np.max(np.abs(soldirect.x - solbrutal.x))
#print np.max(np.abs(soldirect.x - solindirect.x))
#print np.max(np.abs(soldirect.x - soldd.x))

#solindirect = computation.solution(psi.IndirectSolver().solve)
#print solindirect.x[0:20]

#pos.standardoutput(computation, soldirect, 20, bounds, npoints, 'soundsoft')
