import pypwdg.core.bases as pcb
import pypwdg.mesh.mesh as pmm
import pypwdg.core.boundary_data as pcbd
import pypwdg.setup.problem as psp
import pypwdg.setup.computation as psc
import pypwdg.setup.indirect as psi
import pypwdg.core.physics as pcp
import pypwdg.output.solution as pos
import pypwdg.test.utils.mesh as tum
import numpy as np
np.set_printoptions(threshold='nan')

import pypwdg.parallel.main

from numpy import array,sqrt

k = 10
n = 2
g = pcb.FourierHankel([-1,-1], [0], k)
bdytag = "BDY"
bnddata={bdytag:pcbd.dirichlet(g)}

bounds=array([[0,1],[0,1]],dtype='d')
npoints=array([200,200])
mesh = tum.regularsquaremesh(n, bdytag)
#print mesh.nodes
#print mesh.elements
    
problem = psp.Problem(mesh, k, bnddata)
computation = psc.Computation(problem, pcb.planeWaveBases(2,k,5), pcp.HelmholtzSystem, 13)

#solbrutal = computation.solution(psi.BrutalSolver(np.complex, psi.DomainDecompOperator(mesh)).solve)
soldd = computation.solution(psi.IndirectSolver(np.complex, psi.DomainDecompOperator(mesh)).solve)
solindirect = computation.solution(psi.IndirectSolver(np.complex, psi.BlockPrecondOperator(mesh)).solve)
soldirect = computation.solution(psc.DirectSolver().solve)

#print soldirect.x
#print solbrutal.x
#print np.abs(soldirect.x - solbrutal.x)
#print np.max(np.abs(soldirect.x - solbrutal.x))
print np.max(np.abs(soldirect.x - solindirect.x))
print np.max(np.abs(soldirect.x - soldd.x))

#solindirect = computation.solution(psi.IndirectSolver().solve)
#print solindirect.x[0:20]

#pos.standardoutput(computation, soldirect, 20, bounds, npoints, 'soundsoft')
