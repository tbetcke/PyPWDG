import pypwdg.core.bases as pcb
import pypwdg.core.bases.variable as pcbv
import pypwdg.mesh.mesh as pmm
import pypwdg.core.boundary_data as pcbd
import pypwdg.setup.problem as psp
import pypwdg.setup.computation as psc
import pypwdg.setup.indirect as psi
import pypwdg.core.physics as pcp
import pypwdg.output.solution as pos
import pypwdg.test.utils.mesh as tum
import pypwdg.output.mploutput as pom


import pypwdg.parallel.main



import numpy as np

k = 40
#direction=np.array([[1.0,1.0]])/np.sqrt(2)
#g = pcb.PlaneWaves(direction, k)
g = pcb.FourierHankel([-0.5,-0.5], [0], k)
impbd = pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g)


#bnddata={7:pcbd.dirichlet(g), 
#         8:pcbd.dirichlet(g)}

bounds=np.array([[0,1],[0,1]],dtype='d')
npoints=np.array([500,500])

n = 6

bnddata={'BDY':impbd}
mesh = tum.regularsquaremesh(n, 'BDY')

quadpoints = 10
npws = []
conds = []
errs = []
problem = psp.Problem(mesh, k, bnddata)
for npw in range(7,45,2):
    quadpoints = npw
    # Original basis:
    pw = pcbv.PlaneWaveVariableN(pcb.uniformdirs(2,npw))
    basisrule = pw
    
    computation = psc.Computation(psc.ComputationInfo(problem, basisrule, quadpoints),pcp.HelmholtzSystem) 
    solution = computation.solution(psc.DirectOperator(), psc.DirectSolver())
    op = psc.DirectOperator()
    op.setup(computation.system, [], {})
    M = op.mass().todense()
    cond = np.max([np.linalg.cond(M[i:i+npw, i:i+npw]) for i in range(0, M.shape[0], npw)])
    err = pos.comparetrue(bounds, npoints, g, solution)
    conds.append(cond)
    errs.append(err)
    npws.append(npw)
    print npw, err, cond

print npws
print errs
print conds    
#solind = computation.solution(psi.DefaultOperator(), psi.GMRESSolver(dtype='ctor'))

#pom.output2dsoln(bounds, solution, npoints)
#pos.standardoutput(solution, quadpoints, bounds, npoints, 'square', mploutput = True)
