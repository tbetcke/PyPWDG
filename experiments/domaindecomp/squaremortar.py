import pypwdg.core.bases as pcb
import pypwdg.core.bases.reference as pcbr
import pypwdg.mesh.mesh as pmm
import pypwdg.mesh.submesh as pmsm
import pypwdg.core.boundary_data as pcbd
import pypwdg.setup.mortar as psm
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
k = 5
n = 5
g = pcb.FourierHankel([-1,-1], [0], k)
g = pcb.ConstantBasis()
bdytag = "BDY"
bnddata={bdytag:pcbd.dirichlet(g)}

bounds=array([[0,1],[0,1]],dtype='d')
npoints=array([200,200])
mesh = tum.regularsquaremesh(n, bdytag)
problem = psp.Problem(mesh, k, bnddata)
basisrule = pcb.planeWaveBases(2,k,9)
#basisrule = pcbr.ReferenceBasisRule(pcbr.Dubiner(3))
mortarrule = pcbr.ReferenceBasisRule(pcbr.Legendre1D(5))
tracebc = [-2j*k,0]
#tracebc = [0,0]

mc = psm.MortarComputation(problem, basisrule, mortarrule, nquad, pcp.HelmholtzSystem, pcp.HelmholtzBoundary, tracebc)
sol = mc.solution(psm.BrutalSolver(np.complex).solve, dovolumes=True)
#print sol.x
pos.standardoutput(sol, 20, bounds, npoints, 'squaremortar', mploutput=True)


