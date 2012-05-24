import pypwdg.core.bases as pcb
import pypwdg.core.bases.reference as pcbr
import pypwdg.core.boundary_data as pcbd
import pypwdg.setup.mortar as psm
import pypwdg.setup.problem as psp
import pypwdg.setup.computation as psc
import pypwdg.setup.indirect as psi
import pypwdg.mesh.mesh as pmm
import pypwdg.core.physics as pcp
import pypwdg.output.solution as pos
import pypwdg.output.mploutput as pom

import pypwdg.parallel.main


import pypwdg.test.utils.mesh as tum
import numpy as np
np.set_printoptions(threshold='nan')

from numpy import array,sqrt

import matplotlib.pyplot as mp


#nparts = 3
nquad = 5
k = 6
n = 6
g = pcb.FourierHankel([-1,-1], [0], k)
#g = pcb.PlaneWaves([3.0/5,-4.0/5], k)
c = pcb.ConstantBasis()
dg = pcbd.dirichlet(g)
ig = pcbd.generic_boundary_data([-1j*k, 1], [-1j*k, 1], g)
ic = pcbd.generic_boundary_data([-1j*k, 1], [1, 0], c)
bdytag = "BDY"
bnddata={1:dg,2:dg,3:dg,4:dg}

bounds=array([[0,1],[0,1]],dtype='d')
npoints=array([200,200])
#mesh = tum.regularsquaremesh(n, bdytag)
meshinfo = tum.regularrectmeshinfo([0,1], [0,1], n, n)
topology = pmm.Topology(meshinfo)

#def partitions(nparts):
#    ne = meshinfo.nelements
#    return [np.arange((i * ne) / nparts, ((i+1) * ne) / nparts) for i in range(nparts)] 
#
#partition = pmm.BespokePartition(meshinfo, topology, partitions)

#mesh = pmm.MeshView(meshinfo, topology, partition)
mesh = pmm.meshFromInfo(meshinfo)

problem = psp.Problem(mesh, k, bnddata)
basisrule = pcb.planeWaveBases(2,k,7)
#basisrule = pcbr.ReferenceBasisRule(pcbr.Dubiner(0))
mortarrule = pcbr.ReferenceBasisRule(pcbr.Legendre1D(3))
s = -1j*k
#s = 0
#tracebc = [0,0]

mc = psm.MortarComputation(problem, basisrule, mortarrule, nquad, pcp.HelmholtzSystem, pcp.HelmholtzBoundary, s)
sol = mc.solution(psi.BrutalSolver(np.complex), dovolumes=True)
solfaked = mc.fakesolution(g, [s, 1])
#print sol.x
#pos.standardoutput(sol, 20, bounds, npoints, 'squaremortar')
pom.output2dsoln(bounds, sol, npoints, plotmesh = True, show = False)
pom.output2dsoln(bounds, solfaked, npoints, plotmesh = True, show = False)
pom.output2dfn(bounds, g.values, npoints, show=False)

rectmesh = tum.regularrectmesh([0,0.5], [0,1.0], n/2, n)
rectbd = {1:ig, 2:ig, 3:ig, 4:ig}
rectprob = psp.Problem(rectmesh, k, rectbd)
rectcmp = psc.DirectComputation(rectprob, basisrule, nquad, pcp.HelmholtzSystem)
rectsol = rectcmp.solution()
pom.output2dsoln([[0,0.5],[0,1]],rectsol, npoints, plotmesh=True, show=False)

mp.show()
#
#sold = psc.DirectComputation(problem, basisrule, nquad, pcp.HelmholtzSystem).solution()
#pos.standardoutput(sold, 20, bounds, npoints, 'squaremortar', mploutput=True)


