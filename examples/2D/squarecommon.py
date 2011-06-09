import pypwdg.core.bases as pcb
import pypwdg.core.bases.reference as pcbr
import pypwdg.core.bases.variable as pcbv
import pypwdg.mesh.mesh as pmm
import pypwdg.core.boundary_data as pcbd
import pypwdg.setup.problem as psp
import pypwdg.setup.computation as psc
import pypwdg.core.physics as pcp
import pypwdg.output.solution as pos

import pypwdg.core.bases.reduced as pcbred
import pypwdg.utils.quadrature as puq

import pypwdg.parallel.main



import numpy as np

k = 20
direction=np.array([[1.0,1.0]])/np.sqrt(2)
#g = pcb.PlaneWaves(direction, k)
g = pcb.FourierHankel([-2,-2], [0], k)
impbd = pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g)

#bnddata={7:impbd, 
#         8:impbd}
bnddata={7:pcbd.dirichlet(g), 
         8:pcbd.dirichlet(g)}

bounds=np.array([[0,1],[0,1]],dtype='d')
npoints=np.array([500,500])

mesh = pmm.gmshMesh('square.msh',dim=2)
print mesh.nelements

npw = 15
quadpoints = 15

# Original basis:
basisrule = pcbv.PlaneWaveVariableN(pcb.uniformdirs(2,npw))
# Polynomials only:
#basisrule = pcbr.ReferenceBasisRule(pcbr.Dubiner(3))
# Product basis:
#basisrule = pcb.ProductBasisRule(pcb.planeWaveBases(2,k,npw), pcbr.ReferenceBasisRule(pcbr.Dubiner(1)))

basisrule = pcbred.SVDBasisReduceRule(puq.trianglequadrature(quadpoints), basisrule)

problem = psp.Problem(mesh, k, bnddata)
computation = psc.Computation(problem, basisrule, pcp.HelmholtzSystem, quadpoints)
solution = computation.solution(psc.DirectSolver().solve)

pos.comparetrue(bounds, npoints, g, solution)
pos.standardoutput(computation, solution, quadpoints, bounds, npoints, 'square')
