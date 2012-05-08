import pypwdg.core.bases as pcb
import pypwdg.core.bases.reference as pcbr
import pypwdg.mesh.mesh as pmm
import pypwdg.core.boundary_data as pcbd
import pypwdg.core.bases.utilities as pcbu
import pypwdg.setup.problem as psp
import pypwdg.setup.computation as psc
import pypwdg.setup.indirect as psi
import pypwdg.core.physics as pcp
import pypwdg.output.solution as pos
import pypwdg.parallel.decorate as ppd
import pypwdg.test.utils.mesh as tum
import numpy as np
np.set_printoptions(threshold='nan')

from numpy import array,sqrt

@ppd.distribute()
class SkeletonPartition(pmm.Partition):
    def __init__(self, meshview, skelemeshinfo, skelemeshtopology, skelefaceindicator):
        pmm.Partition.__init__(self, skelemeshinfo, skelemeshtopology, meshview.facepartition.diagonal()[skelefaceindicator].nonzero()[0])
        
import pypwdg.parallel.main

def skeletonMesh(mesh, skeletontag):
    ''' Creates a codimension 1 submesh consisting of the faces associated with the boundary highlighted by skeletontag
        Also returns the element->face map from the submesh back to the original mesh
    '''
    
    skeletonfaceindicator = mesh.topology.faceentities == skeletontag  
    skeletonelts = mesh.faces[skeletonfaceindicator]
    meshinfo = pmm.SimplicialMeshInfo(mesh.nodes, skeletonelts, None, {}, mesh.dim -1)
    topology = pmm.Topology(meshinfo)
    partition = SkeletonPartition(mesh, meshinfo, topology, skeletonfaceindicator)
    return pmm.MeshView(meshinfo, topology, partition), skeletonfaceindicator

def skeletonBasis(skelemesh, basisrule):
    etob = {}
    ei = pcbu.ElementInfo(skelemesh, 0)
    for e in skelemesh.partition:
        etob[e] = basisrule.populate(ei.info(e))
    return pcbu.CellToBases(etob)
    

class SkeletonFaceToBasis(object):
    def __init__(self, skeleelttobasis, skeletonfaceindicator):
        nfaces = len(skeletonfaceindicator)
        self.elttobasis = skeleelttobasis
        self.numbases = np.zeros(nfaces, dtype=int)
        self.numbases[skeletonfaceindicator] = skeleelttobasis.getSizes()
        self.indices = np.cumsum(np.concatenate(([0], self.numbases)))
        self.skeletonfaceindicator = skeletonfaceindicator
        self.skeletonfaceindex = np.ones(nfaces,dtype=int) * -1
        self.skeletonfaceindex[skeletonfaceindicator] = skeletonfaceindicator.nonzero()[0]
             
    def evaluate(self, faceid, points):        
        skeletonelt = self.skeletonfaceindex[faceid]
        if skeletonelt >=0: 
            vals = self.elttobasis.getValues(skeletonelt, points)
            derivs = vals
            return (vals,derivs)
        else:
            raise Exception('Bad faceid for Skeleton %s,%s'%(faceid, skeletonelt))

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

cutfaces = ((part.cutfaces==1).nonzero()[0] for part in parts)
cutvertices = list(np.concatenate([meshinfo.faces[fs] for fs in cutfaces]).flatten())

boundaries = list(meshinfo.boundaries)
boundaries.append(('INTERNAL', cutvertices))

meshinfo2 = pmm.SimplicialMeshInfo(meshinfo.nodes, meshinfo.elements, meshinfo.elemIdentity, boundaries, meshinfo.dim)
topology2 = pmm.Topology(meshinfo2)

mortarrule = pcbr.ReferenceBasisRule(pcbr.Legendre1D(2))

class DummySolver(object):
    def solve(self, system, sysargs, syskwargs):
        S,G = system.getSystem(*sysargs, **syskwargs)
        M = S.tocsr()
        print M
        print G

for part in parts:
    mesh = pmm.MeshView(meshinfo2, topology2, part)
    skeletonmesh, skeleindicator = skeletonMesh(mesh, 'INTERNAL')

    skeletob  = skeletonBasis(skeletonmesh, mortarrule)
    skelftob = SkeletonFaceToBasis(skeletob, skeleindicator)
    
    problem = psp.Problem(mesh, k, bnddata)
    problem.bdyinfo['INTERNAL'] = (pcbd.BoundaryCondition([-1j*k, 1], [1, 0]), skelftob)

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
