import pypwdg.core.bases as pcb
import pypwdg.core.bases.reference as pcbr
import pypwdg.core.bases.variable as pcbv
import pypwdg.mesh.mesh as pmm
import pypwdg.core.boundary_data as pcbd
import pypwdg.setup.problem as psp
import pypwdg.setup.computation as psc
import pypwdg.core.physics as pcp
import pypwdg.output.solution as pos
import wavefrontexamples as w
import pypwdg.raytrace.wavefront as prw
import pypwdg.raytrace.basisrules as prb
import pypwdg.output.mploutput as pom
import pypwdg.parallel.main

import numpy as np

k = 40
direction=np.array([[0.0,1.0]])
g = pcb.PlaneWaves(direction, k)
#g = pcb.FourierHankel([-2,-2], [0], k)
impbd = pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g)

bnddata={7:impbd, 
         8:impbd}
#bnddata={7:pcbd.dirichlet(g), 
#         8:pcbd.dirichlet(g)}

bounds=np.array([[0,1],[0,1]],dtype='d')
npoints=np.array([500,500])

mesh = pmm.gmshMesh('../../examples/2D/square.msh',dim=2)
print mesh.nelements

npw = 20
quadpoints = 10

c = 1
N = 20
slow, gradslow = w.bubble(c,0.2,0.3)
entityton = {6:lambda p : 1.0 / slow(p)}
#entityton = {6:1}

x0 = np.vstack((np.linspace(0,1,N),np.zeros(N))).T
p0 = np.vstack((np.zeros(N),np.ones(N))).T
wavefronts, forwardidxs = prw.wavefront(x0, p0, slow, gradslow, 0.1, 1.2/c, 0.1)
vtods = prw.nodesToPhases(wavefronts, forwardidxs, mesh, [7,8])
rt = prb.RaytracedBasisRule(vtods)

# Original basis:
pw = pcbv.PlaneWaveVariableN(pcb.uniformdirs(2,npw))

# Polynomials only:
poly = pcbr.ReferenceBasisRule(pcbr.Dubiner(2))
# Product basis:
#basisrule = pcb.ProductBasisRule(pcb.planeWaveBases(2,k,npw), pcbr.ReferenceBasisRule(pcbr.Dubiner(1)))

#basisrule=pcb.ProductBasisRule(pw,pcbr.ReferenceBasisRule(pcbr.Dubiner(0)))
prod = pcb.ProductBasisRule(rt, poly)
basisrule = prod
#basisrule = pcbred.SVDBasisReduceRule(puq.trianglequadrature(quadpoints), basisrule)

problem = psp.VariableNProblem(entityton, mesh, k, bnddata)

computation = psc.Computation(problem, basisrule, pcp.HelmholtzSystem, quadpoints)
solution = computation.solution(psc.DirectSolver().solve, dovolumes=True)

#pos.comparetrue(bounds, npoints, g, solution)
#pom.output2dsoln(bounds, solution, npoints)
pos.standardoutput(computation, solution, quadpoints, bounds, npoints, mploutput = True)
pom.showdirections(mesh, prb.getetob(wavefronts, forwardidxs, mesh, [7,8]) ,scale=20)
#w.plotwavefront(wavefronts, forwardidxs)
pom.output2dfn(bounds, slow, npoints)