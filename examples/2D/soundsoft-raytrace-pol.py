import pypwdg.setup.problem as psp
import pypwdg.setup.computation as psc
import pypwdg.core.bases as pcb
import pypwdg.mesh.mesh as pmm
import pypwdg.core.bases.variable as pcbv
import pypwdg.core.boundary_data as pcbd
import pypwdg.core.bases.reference as pcbr
import pypwdg.adaptivity.adaptivity2 as paa
import pypwdg.core.physics as pcp
import pypwdg.output.solution as pos
import pypwdg.raytrace.control as prc
import pypwdg.output.basis as pob
import pypwdg.parallel.main

from numpy import array,sqrt
import numpy as np

k = 15
direction=array([[1.0,1.0]])/sqrt(2)
g = pcb.PlaneWaves(direction, k)

bnddata={11:pcbd.zero_dirichlet(),
         10:pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g=g)}

bounds=array([[-2,2],[-2,2]],dtype='d')
npoints=array([200,200])

mesh = pmm.gmshMesh('squarescatt.msh',dim=2)

quadpoints = 20
p=3

entityton ={9:1}
problem=psp.VariableNProblem(entityton, mesh,k, bnddata)
etods = prc.tracemesh(problem, {10:lambda x:direction})

for idx in range(len(etods)):
    if len(etods[idx])==0: 
        etods[idx]=direction
    else:
        tmp=np.dot(etods[idx],direction.T)
        if min(tmp)<1-1E-5: etods[idx]=np.vstack((etods[idx],direction))

etob = [[pcb.PlaneWaves(ds, k)] if len(ds) else [] for ds in etods]
pob.vtkbasis(mesh,etob,'soundsoftrays.vtu',None)

b0=pcbv.PlaneWaveVariableN(pcb.circleDirections(20))
b=pcb.PlaneWaveFromDirectionsRule(etods)
b1=pcb.ProductBasisRule(b,pcbr.ReferenceBasisRule(pcbr.Dubiner(p)))
b2=pcbr.ReferenceBasisRule(pcbr.Dubiner(p))

computation = psc.Computation(problem, b1, pcp.HelmholtzSystem, quadpoints)

solution = computation.solution(psc.DirectSolver().solve, dovolumes=True)
pos.standardoutput(computation, solution, quadpoints, bounds, npoints, 'soundsoft_pol')
print solution.getError('Dirichlet')
print solution.getError('Neumann')
print solution.getError('Boundary')

