import pypwdg.core.bases as pcb
import pypwdg.mesh.mesh as pmm
import pypwdg.core.boundary_data as pcbd
import pypwdg.setup.problem as psp
import pypwdg.setup.computation as psc
import pypwdg.setup.domain as psd
import pypwdg.setup.indirect as psi
import pypwdg.core.physics as pcp
import pypwdg.output.solution as pos
import pypwdg.utils.file as puf
import numpy as np
np.set_printoptions(threshold='nan')

import pypwdg.parallel.main

from numpy import array,sqrt

k = 10
direction=array([[1.0,1.0]])/sqrt(2)
g = pcb.PlaneWaves(direction, k)

bnddata={11:pcbd.zero_dirichlet(),
         10:pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g=g)}

bounds=array([[-2,2],[-2,2]],dtype='d')
npoints=array([200,200])
with puf.pushd('../../examples/2D'):
    mesh = pmm.gmshMesh('squarescatt.msh',dim=2)
    
problem = psp.Problem(mesh, k, bnddata)
compinfo = psc.ComputationInfo(problem, pcb.planeWaveBases(2,k,11), 5)
computation = psc.Computation(compinfo, pcp.HelmholtzSystem)

#solbrutal = computation.solution(psi.BrutalSolver(np.complex, psi.DomainDecompOperator(mesh)).solve)
soldd = computation.solution(psd.SchwarzOperator(mesh), psi.GMRESSolver(np.complex))
solindirect = computation.solution(psi.BlockPrecondOperator(mesh), psi.GMRESSolver(np.complex))
soldirect = computation.solution(psc.DirectOperator(), psc.DirectSolver())

#print np.max(np.abs(soldirect.x - solbrutal.x))
print np.max(np.abs(soldirect.x - solindirect.x))
print np.max(np.abs(soldirect.x - soldd.x))

#solindirect = computation.solution(psi.IndirectSolver().solve)
#print solindirect.x[0:20]

#pos.standardoutput(computation, soldirect, 20, bounds, npoints, 'soundsoft')
