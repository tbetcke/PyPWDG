import pypwdg.core.bases as pcb
import pypwdg.core.bases.reference as pcbr
import pypwdg.core.boundary_data as pcbd
import pypwdg.setup.mortar as psm
import pypwdg.setup.problem as psp
import pypwdg.setup.indirect as psi
import pypwdg.core.physics as pcp
import pypwdg.output.solution as pos
import pypwdg.output.mploutput as pom

import pypwdg.parallel.main


import pypwdg.test.utils.mesh as tum
import numpy as np
np.set_printoptions(threshold='nan')

from numpy import array,sqrt

import matplotlib.pyplot as mp


nparts = 3
nquad = 3
k = 5
n = 4
g = pcb.FourierHankel([-1,-1], [0], k)
#g = pcb.ConstantBasis()
bdytag = "BDY"
bnddata={bdytag:pcbd.dirichlet(g)}

bounds=array([[0,1],[0,1]],dtype='d')
npoints=array([200,200])
mesh = tum.regularsquaremesh(n, bdytag)
problem = psp.Problem(mesh, k, bnddata)
basisrule = pcb.planeWaveBases(2,k,9)
#basisrule = pcbr.ReferenceBasisRule(pcbr.Dubiner(0))
mortarrule = pcbr.ReferenceBasisRule(pcbr.Legendre1D(2))
tracebc = [-2j*k,0]
#tracebc = [0,0]

mc = psm.MortarComputation(problem, basisrule, mortarrule, nquad, pcp.HelmholtzSystem, pcp.HelmholtzBoundary, tracebc)
sol = mc.solution(psi.BrutalSolver(np.complex), dovolumes=True)
solfaked = mc.fakesolution(g, [-1j*k, 1])
#print sol.x
pos.standardoutput(sol, 20, bounds, npoints, 'squaremortar')
pom.output2dsoln(bounds, sol, npoints, plotmesh = True, show = False)
pom.output2dsoln(bounds, solfaked, npoints, plotmesh = True, show = False)
mp.show()
#
#sold = psc.DirectComputation(problem, basisrule, nquad, pcp.HelmholtzSystem).solution()
#pos.standardoutput(sold, 20, bounds, npoints, 'squaremortar', mploutput=True)


