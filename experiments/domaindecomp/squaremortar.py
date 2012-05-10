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
import pypwdg.mesh.meshutils as pmmu
import pypwdg.utils.quadrature as puq
import pypwdg.core.assembly as pca
import pypwdg.core.vandermonde as pcv
import pypwdg.mesh.structure as pms

import pypwdg.test.utils.mesh as tum
import numpy as np
np.set_printoptions(threshold='nan')

from numpy import array,sqrt

@ppd.distribute()
class SkeletonPartition(pmm.Partition):
    def __init__(self, meshview, skelemeshinfo, skelemeshtopology, skelefacemap):
        pmm.Partition.__init__(self, skelemeshinfo, skelemeshtopology, skelefacemap.partition())
        
class SkeletonFaceMap(object):
    
    def __init__(self, mesh, skeletontag):
        print "SkeletonFaceMap.init"
        self.indicator = mesh.topology.faceentities==skeletontag
        self.index = np.ones(mesh.nfaces, dtype=int) * -1
        self.index[self.indicator] = np.arange(sum(self.indicator))
        self.skeltomeshindex = self.indicator.nonzero()[0]
        self.mesh = mesh
        
    def expand(self, skeleeltarray):
        meshfacearray = np.zeros(len(self.indicator), dtype=skeleeltarray.dtype)
        meshfacearray[self.indicator] = skeleeltarray
        return meshfacearray
    
    def partition(self):
        return self.mesh.facepartition.diagonal()[self.indicator].nonzero()[0]

                        
import pypwdg.parallel.main


def skeletonMesh(mesh, skeletontag):
    ''' Creates a codimension 1 submesh consisting of the faces associated with the boundary highlighted by skeletontag
        Also returns the element->face map from the submesh back to the original mesh
    '''
    skeletonfacemap = SkeletonFaceMap(mesh, skeletontag)
    skeletonelts = mesh.faces[skeletonfacemap.indicator]
    meshinfo = pmm.SimplicialMeshInfo(mesh.nodes, skeletonelts, None, {}, mesh.dim -1)
    topology = pmm.Topology(meshinfo)
    partition = SkeletonPartition(mesh, meshinfo, topology, skeletonfacemap)
    return pmm.MeshView(meshinfo, topology, partition), skeletonfacemap

def skeletonBasis(skelemesh, basisrule):
    etob = {}
    ei = pcbu.ElementInfo(skelemesh, 0)
    print "skeletonBasis"
    for e in skelemesh.partition:
        print e
        etob[e] = basisrule.populate(ei.info(e))
    return pcbu.CellToBases(etob, np.arange(skelemesh.nelements))

class SkeletonFaceToBasis(object):
    def __init__(self, skeleelttobasis, skeletonfacemap):
        self.elttobasis = skeleelttobasis
        self.skeletonfacemap = skeletonfacemap
             
    def evaluate(self, faceid, points):        
        print "SkeletonFaceToBasis.evaluate %s"%faceid
        skeletonelt = self.skeletonfacemap.index[faceid]
        print skeletonelt
        if skeletonelt >=0: 
            vals = self.elttobasis.getValues(skeletonelt, points)
            derivs = vals
            print derivs.shape
            return (vals,derivs)
        else:
            raise Exception('Bad faceid for Skeleton %s,%s'%(faceid, skeletonelt))
    
    @property
    def numbases(self):
        return self.skeletonfacemap.expand(self.elttobasis.getSizes())
    
    @property
    def indices(self):
        return np.cumsum(np.concatenate(([0], self.numbases)))

nparts = 3
nquad = 10
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
Bs = []
for part in parts:
    mesh = pmm.MeshView(meshinfo2, topology2, part)
    skeletonmesh, skeletonfacemap = skeletonMesh(mesh, 'INTERNAL')
    print skeletonmesh.partition
    print skeletonmesh.directions.shape, mesh.directions.shape

    skeletob  = skeletonBasis(skeletonmesh, mortarrule)
    skelftob = SkeletonFaceToBasis(skeletob, skeletonfacemap)
    
    problem = psp.Problem(mesh, k, bnddata)
    compinfo = psc.ComputationInfo(problem, pcb.planeWaveBases(2,k,5), nquad)
    system = pcp.HelmholtzSystem(compinfo)
    
    AA,G = system.getSystem()
    A = AA.tocsr()
    print A
    print G
    B = system.getBoundary('INTERNAL', (pcbd.BoundaryCoefficients([-1j*k, 1], [1, 0]), skelftob))
    print B.load(False).tocsr()
    print B.stiffness().tocsr()
    Bs.append(B.load(False).tocsr())
    
    fquad, equad = puq.quadrules(skeletonmesh.dim, nquad)
    elementquads = pmmu.MeshElementQuadratures(skeletonmesh, equad)
    ev = pcv.ElementVandermondes(skeletonmesh, skeletob, elementquads)
    massassembly = pca.Assembly(ev, ev, elementquads.quadweights)
    E = pms.ElementMatrices(skeletonmesh)
    M = massassembly.assemble([[E.I, E.Z],[E.Z, E.Z]])
    print M.tocsr()
    
    

    
    
    
##print "DD solve"
##soldd = computation.solution(psi.IndirectSolver(np.complex, psi.DomainDecompOperator(mesh)).solve)
##print "Block precond solve"
##solindirect = computation.solution(psi.IndirectSolver(np.complex, psi.BlockPrecondOperator(mesh)).solve)
#print "Direct solve"
#soldirect = computation.solution(psc.DirectSolver().solve)
#
##print soldirect.x
##print solbrutal.x
##print np.abs(soldirect.x - solbrutal.x)
#print np.max(np.abs(soldirect.x - solbrutal.x))
##print np.max(np.abs(soldirect.x - solindirect.x))
##print np.max(np.abs(soldirect.x - soldd.x))
#
##solindirect = computation.solution(psi.IndirectSolver().solve)
##print solindirect.x[0:20]

#pos.standardoutput(computation, soldirect, 20, bounds, npoints, 'soundsoft')
