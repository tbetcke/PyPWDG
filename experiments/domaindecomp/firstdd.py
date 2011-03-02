import pypwdg.setup as ps
import pypwdg.core.bases as pcb
import pypwdg.mesh.mesh as pmm
import pypwdg.core.boundary_data as pcbd
import pypwdg.core.evaluation as pce
from numpy import array,sqrt,vstack,ones,linspace
import pylab as pl

k = 10
direction=array([[1.0,1.0]])/sqrt(2)
g = pcb.PlaneWaves(direction, k)
impbd = pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g)

bnddata1={7:pcbd.zero_neumann(), 
         8:pcbd.dirichlet(g),
         9:pcbd.dirichlet(g),
         10:pcbd.dirichlet(g)}

bounds=array([[0,1],[0,1]],dtype='d')
npoints=array([1000,1000])

mesh = pmm.gmshMesh('square.msh',dim=2)
bases = pcb.planeWaveBases(mesh,k,nplanewaves=30)

problem1 = ps.Problem(mesh,k,20, bnddata1)
solution1 = ps.Computation(problem1, bases).solve()

points = vstack([ones(1000), linspace(0, 1, 1000)]).transpose()
#print points
evalu = pce.Evaluator(mesh, solution1.elttobasis, array(points), direction=(1, -1))
vals, derivs = evalu.evaluate(solution1.x)

solution1.writeSolution(bounds,npoints,fname='firstdd.vti')
problem1.writeMesh(fname='firstdd.vtu',scalars=solution1.combinedError())

pl.plot (vals)
pl.plot (derivs/sqrt(2) * 1./k)
pl.show()

