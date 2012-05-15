import pypwdg.core.bases as pcb
import pypwdg.core.bases.reference as pcbr
import pypwdg.mesh.mesh as pmm
import pypwdg.mesh.submesh as pmsm
import pypwdg.core.boundary_data as pcbd
import pypwdg.setup.domain as psd
import pypwdg.setup.problem as psp
import pypwdg.setup.computation as psc
import pypwdg.setup.indirect as psi
import pypwdg.core.physics as pcp
import pypwdg.output.solution as pos
import pypwdg.mesh.meshutils as pmmu
import pypwdg.utils.quadrature as puq
import pypwdg.core.assembly as pca
import pypwdg.core.vandermonde as pcv
import pypwdg.mesh.structure as pms

import pypwdg.parallel.main


import pypwdg.test.utils.mesh as tum
import numpy as np
np.set_printoptions(threshold='nan')

from numpy import array,sqrt

import matplotlib.pyplot as mp


nparts = 3
nquad = 10
k = 10
n = 5
g = pcb.FourierHankel([-1,-1], [0], k)
bdytag = "BDY"
bnddata={bdytag:pcbd.dirichlet(g)}

bounds=array([[0,1],[0,1]],dtype='d')
npoints=array([200,200])
mesh = tum.regularsquaremesh(n, bdytag)
problem = psp.Problem(mesh, k, bnddata)
basisrule = pcb.planeWaveBases(2,k,7)
mortarrule = pcbr.ReferenceBasisRule(pcbr.Legendre1D(2))
tracebc = [1j*k,0]

mc = psd.MortarComputation(problem, basisrule, mortarrule, nquad, pcp.HelmholtzSystem, tracebc)
sol = mc.solution(psd.BrutalSolver(np.complex).solve)

#
#sd = pmsm.SkeletonisedDomain(meshinfo, 'INTERNAL')
##
##mesh = pmm.meshFromInfo(meshinfo)
##
##cutvertices = pmsm.getCutVertices(mesh)
##
##boundaries = list(meshinfo.boundaries)
##boundaries.append(('INTERNAL', cutvertices))
##
##meshinfo2 = pmm.SimplicialMeshInfo(meshinfo.nodes, meshinfo.elements, meshinfo.elemIdentity, boundaries, meshinfo.dim)
##topology2 = pmm.Topology(meshinfo2)
#
#mortarrule = pcbr.ReferenceBasisRule(pcbr.Legendre1D(2))
#skeletob = psp.constructBasisOnMesh(sd.skeletonmesh, mortarrule) 
#skelftob = psd.SkeletonFaceToBasis(skeletob, sd)
#
#    
#problem = psp.Problem(sd.mesh, k, bnddata)
#compinfo = psc.ComputationInfo(problem, pcb.planeWaveBases(2,k,5), nquad)
#system = pcp.HelmholtzSystem(compinfo)
#
#AA,G = system.getSystem()
#A = AA.tocsr()
##    print A
##    print G
#boundary = pcp.HelmholtzBoundary(compinfo, 'INTERNAL', (pcbd.BoundaryCoefficients([-1j*k, 1], [1, 0]), skelftob))
##    print B.load(False).tocsr()
##    print B.stiffness().tocsr()
#B = boundary.load(False).tocsr()
#
#mp.spy(A, markersize = 1)
#mp.figure()
#mp.spy(B, markersize = 1)
#mp.show()
#
##
#Bs = []
#for part in parts:
#    mesh = pmm.MeshView(meshinfo2, topology2, part)
#    skeletonmesh, skeletonfacemap = skeletonMesh(mesh, 'INTERNAL')
#    print skeletonmesh.partition
#    print skeletonmesh.directions.shape, mesh.directions.shape
#
#    skeletob  = skeletonBasis(skeletonmesh, mortarrule)
#    skelftob = SkeletonFaceToBasis(skeletob, skeletonfacemap)
#    
#    problem = psp.Problem(mesh, k, bnddata)
#    compinfo = psc.ComputationInfo(problem, pcb.planeWaveBases(2,k,5), nquad)
#    system = pcp.HelmholtzSystem(compinfo)
#    
#    AA,G = system.getSystem()
#    A = AA.tocsr()
##    print A
##    print G
#    B = system.getBoundary('INTERNAL', (pcbd.BoundaryCoefficients([-1j*k, 1], [1, 0]), skelftob))
##    print B.load(False).tocsr()
##    print B.stiffness().tocsr()
#    Bs.append(B.load(False).tocsr())
#    
#    fquad, equad = puq.quadrules(skeletonmesh.dim, nquad)
#    elementquads = pmmu.MeshElementQuadratures(skeletonmesh, equad)
#    ev = pcv.ElementVandermondes(skeletonmesh, skeletob, elementquads)
#    massassembly = pca.Assembly(ev, ev, elementquads.quadweights)
#    E = pms.ElementMatrices(skeletonmesh)
#    M = massassembly.assemble([[E.I, E.Z],[E.Z, E.Z]])
#    print M.tocsr().shape
#    
#    

    
    
    
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
