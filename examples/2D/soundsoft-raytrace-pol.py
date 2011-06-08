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

k = 25
direction=array([[1.0,1.0]])/sqrt(2)
g = pcb.PlaneWaves(direction, k)

bnddata={11:pcbd.zero_dirichlet(),
         10:pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g=g)}

bounds=array([[-2,2],[-2,2]],dtype='d')
npoints=array([200,200])

mesh = pmm.gmshMesh('squarescatt.msh',dim=2)

quadpoints = 20
p=2
beta=(1.0*p**2)/.2
alpha=1.0/.2
delta=1./.2

entityton ={9:1}
problem=psp.VariableNProblem(entityton, mesh,k, bnddata)
etods = prc.tracemesh(problem, {10:lambda x:direction})


etob = [[pcb.PlaneWaves(ds, k)] if len(ds) else [] for ds in etods]
pob.vtkbasis(mesh,etob,'soundsoftrays.vtu',None)

b0=pcbv.PlaneWaveVariableN(pcb.circleDirections(20))
b=pcb.PlaneWaveFromDirectionsRule(etods)
origins=np.array([[-.5,-.5],[-.5,.5],[.5,-.5],[.5,.5]])
h=pcb.FourierHankelBasisRule(origins,[0])
h2=pcb.ProductBasisRule(h,pcbr.ReferenceBasisRule(pcbr.Dubiner(p)))

b1=pcb.ProductBasisRule(b,pcbr.ReferenceBasisRule(pcbr.Dubiner(p)))
bh=pcb.UnionBasisRule([h2,b1])


b2=pcbr.ReferenceBasisRule(pcbr.Dubiner(p))

computation = psc.Computation(problem, bh, pcp.HelmholtzSystem, quadpoints,alpha=alpha,beta=beta,delta=delta)
solution = computation.solution(psc.DirectSolver().solve, dovolumes=True)
pos.standardoutput(computation, solution, quadpoints, bounds, npoints, 'soundsoft_pol')
print solution.getError('Dirichlet')
print solution.getError('Neumann')
print solution.getError('Boundary')

