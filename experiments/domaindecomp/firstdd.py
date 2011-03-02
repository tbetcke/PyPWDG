import pypwdg.setup as ps
import pypwdg.core.bases as pcb
import pypwdg.mesh.mesh as pmm
import pypwdg.core.boundary_data as pcbd
import pypwdg.core.evaluation as pce
from numpy import array,sqrt,vstack,ones,linspace
import pylab as pl

k = 60
direction=array([[1.0,1.0]])/sqrt(2)
g = pcb.PlaneWaves(direction, k)
impbd = pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g)

bnddata={7:pcbd.dirichlet(g), 
         8:pcbd.dirichlet(g),
         9:pcbd.dirichlet(g),
         10:pcbd.dirichlet(g)}

bounds=array([[0,1],[0,1]],dtype='d')
npoints=array([100,100])

mesh = pmm.gmshMesh('square.msh',dim=2)
bases = pcb.planeWaveBases(mesh,k,nplanewaves=15)

problem=ps.Problem(mesh,k,20, bnddata)
solution = ps.Computation(problem, bases).solve()

points = vstack([linspace(0, 1, 1000), ones(1000)]).transpose()
evalu = pce.Evaluator(mesh, solution.elttobasis, array(points), direction=(1, -1))
vals, derivs = evalu.evaluate(solution.x)

pl.plot (vals)
pl.plot (derivs)
pl.show()

