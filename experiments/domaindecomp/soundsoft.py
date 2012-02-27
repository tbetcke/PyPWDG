import pypwdg.core.bases as pcb
import pypwdg.mesh.mesh as pmm
import pypwdg.core.boundary_data as pcbd
import pypwdg.setup.problem as psp
import pypwdg.setup.computation as psc
import pypwdg.setup.indirect as psi
import pypwdg.core.physics as pcp
import pypwdg.output.solution as pos
import pypwdg.utils.file as puf

import pypwdg.parallel.main

import numpy as np
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
computation = psc.Computation(problem, pcb.planeWaveBases(2,k,13), pcp.HelmholtzSystem, 5)

solindirect = computation.solution(psi.IndirectSolver(np.complex, mesh).solve)
soldirect = computation.solution(psc.DirectSolver().solve)

print np.max(np.abs(solindirect.x - solindirect.x))

sm = psi.SystemMultiply(computation.system, [5], {})
b = sm.multiply(soldirect.x).squeeze()
print np.max(np.abs(b-sm.getRHS().squeeze()))

#solindirect = computation.solution(psi.IndirectSolver().solve)
#print solindirect.x[0:20]

#pos.standardoutput(computation, soldirect, 20, bounds, npoints, 'soundsoft')
