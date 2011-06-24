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

import pypwdg.core.bases.reduced as pcbred
import pypwdg.utils.quadrature as puq


from numpy import array,sqrt
import numpy as np

class bndfun(object):
    def __init__(self):
        pass
    
    def values(self,x):
        return np.ones((x.shape[0],1))
    
    def derivs(self,x,n=None):
        return np.zeros((x.shape[0],1))



def nfun(p):
    #return 1+.5*1./4*(p[:,0]-1)
    return np.exp(1./5*p[:,0]*(5-p[:,0]))
    
import pypwdg.parallel.main

k = 10
direction=array([[1.0,0]])
g = pcb.PlaneWaves(direction, k)




bnddata={7:pcbd.dirichlet(g),
         8:pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g=g)}

bnddata={7:pcbd.dirichlet(g), 8:pcbd.zero_impedance(k)}

bounds=array([[0.001,4.999],[0,1]],dtype='d')
npoints=array([500,100])

mesh = pmm.gmshMesh('waveguide.msh',dim=2)


quadpoints = 30
p=3
t=p**2
g=2
h=.1

beta=p**2*h*k
alpha=k*p**2*1./h
delta=.5

alpha=.5
beta=.5
delta=.5

entityton ={6:nfun}
problem=psp.VariableNProblem(entityton, mesh,k, bnddata)
etods = prc.tracemesh(problem, {8:lambda x:direction})


etob = [[pcb.PlaneWaves(ds, k)] if len(ds) else [] for ds in etods]
pob.vtkbasis(mesh,etob,'waveguide_rays.vtu',None)

b0=pcbv.PlaneWaveVariableN(pcb.circleDirections(10))
b=pcb.PlaneWaveFromDirectionsRule(etods)

b1=pcb.ProductBasisRule(b,pcbr.ReferenceBasisRule(pcbr.Dubiner(p)))
b2=pcbr.ReferenceBasisRule(pcbr.Dubiner(p))

origins=np.array([[0,0],[0,5],[5,1],[0,1]])
h=pcb.FourierHankelBasisRule(origins,[0])
h2=pcb.ProductBasisRule(h,pcbr.ReferenceBasisRule(pcbr.Dubiner(p)))
bh=pcb.UnionBasisRule([h2,b1])
b3=pcb.ProductBasisRule(b0,b2)


basisrule = b3 #cbred.SVDBasisReduceRule(puq.trianglequadrature(quadpoints), bh, threshold=1E-5)


computation = psc.Computation(problem, basisrule, pcp.HelmholtzSystem, quadpoints,alpha=alpha,beta=beta,delta=delta)
solution = computation.solution(psc.DirectSolver().solve, dovolumes=True)



pos.standardoutput(computation, solution, quadpoints, bounds, npoints, 'waveguide')
print solution.getError('Dirichlet')
print solution.getError('Neumann')
print solution.getError('Boundary')

