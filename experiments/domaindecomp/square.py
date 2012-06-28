import pypwdg.core.bases as pcb
import pypwdg.core.bases.reference as pcbr
import pypwdg.core.boundary_data as pcbd
import pypwdg.setup.problem as psp
import pypwdg.setup.computation as psc
import pypwdg.setup.indirect as psi
import pypwdg.setup.mortar as psm
import pypwdg.setup.domain as psd
import pypwdg.core.physics as pcp
import pypwdg.output.solution as pos

import pypwdg.test.utils.mesh as tum
import numpy as np
np.set_printoptions(threshold='nan')

import pypwdg.parallel.main

from numpy import array,sqrt

k = 20
n = 6
g = pcb.FourierHankel([-1,-1], [0], k)
bdytag = "BDY"
bnddata={bdytag:pcbd.dirichlet(g)}

bounds=array([[0,1],[0,1]],dtype='d')
npoints=array([200,200])
mesh = tum.regularsquaremesh(n, bdytag)
#print mesh.nodes
#print mesh.elements
    
problem = psp.Problem(mesh, k, bnddata)
basisrule = pcb.planeWaveBases(2,k,9)
nquad = 10
system = pcp.HelmholtzSystem
compinfo = psc.ComputationInfo(problem, basisrule, 10)
computation = psc.Computation(compinfo, system)

solblockdiagonal = computation.solution(psi.DiagonalBlockOperator(mesh), psi.GMRESSolver('ctor'))
soldomaindecomp = computation.solution(psd.DomainDecompOperator(mesh), psi.GMRESSolver('ctor'))
mortarrule = pcbr.ReferenceBasisRule(pcbr.Legendre1D(3))
s = -1j*k
mc = psm.MortarComputation(problem, basisrule, mortarrule, nquad, system, system.boundaryclass, s)
solmortar = mc.solution(psi.GMRESSolver('ctor'))


#solbrutal = computation.solution(psd.DomainDecompOperator(mesh), psi.GMRESSolver('ctor'))
#print "DD solve"
#soldd = computation.solution(psi.IndirectSolver(np.complex, psi.DomainDecompOperator(mesh)).solve)
#print "Block precond solve"
#solindirect = computation.solution(psi.IndirectSolver(np.complex, psi.BlockPrecondOperator(mesh)).solve)
print "Direct solve"
soldirect = computation.solution(psc.DirectOperator(), psc.DirectSolver())

#print soldirect.x
#print solbrutal.x
#print np.abs(soldirect.x - solbrutal.x)
#print np.max(np.abs(soldirect.x - solbrutal.x))
print np.max(np.abs(soldirect.x - solblockdiagonal.x))
#print np.max(np.abs(soldirect.x - solindirect.x))
#print np.max(np.abs(soldirect.x - soldd.x))

#solindirect = computation.solution(psi.IndirectSolver().solve)
#print solindirect.x[0:20]

#pos.standardoutput(soldirect, 20, bounds, npoints, 'square', mploutput = True)
