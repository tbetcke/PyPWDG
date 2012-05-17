import pypwdg.core.bases as pcb
import pypwdg.core.boundary_data as pcbd
import pypwdg.setup.problem as psp
import pypwdg.setup.computation as psc
import pypwdg.setup.indirect as psi
import pypwdg.setup.domain as psd
import pypwdg.core.physics as pcp
import pypwdg.output.solution as pos

import pypwdg.test.utils.mesh as tum
import numpy as np
np.set_printoptions(threshold='nan')

import pypwdg.parallel.main

from numpy import array,sqrt

k = 10
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
compinfo = psc.ComputationInfo(problem, pcb.planeWaveBases(2,k,9), 13)
computation = psc.Computation(compinfo, pcp.HelmholtzSystem)
solbrutal = computation.solution(psd.DomainDecompOperator(mesh), psi.BrutalSolver(np.complex))
#print "DD solve"
#soldd = computation.solution(psi.IndirectSolver(np.complex, psi.DomainDecompOperator(mesh)).solve)
#print "Block precond solve"
#solindirect = computation.solution(psi.IndirectSolver(np.complex, psi.BlockPrecondOperator(mesh)).solve)
print "Direct solve"
soldirect = computation.solution(psc.DirectOperator(), psc.DirectSolver())

#print soldirect.x
#print solbrutal.x
#print np.abs(soldirect.x - solbrutal.x)
print np.max(np.abs(soldirect.x - solbrutal.x))
#print np.max(np.abs(soldirect.x - solindirect.x))
#print np.max(np.abs(soldirect.x - soldd.x))

#solindirect = computation.solution(psi.IndirectSolver().solve)
#print solindirect.x[0:20]

pos.standardoutput(soldirect, 20, bounds, npoints, 'square', mploutput = True)
