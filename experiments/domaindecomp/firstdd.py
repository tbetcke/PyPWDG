import pypwdg.setup as ps
import pypwdg.core.bases as pcb
import pypwdg.mesh.mesh as pmm
import pypwdg.core.boundary_data as pcbd
import pypwdg.core.evaluation as pce
from numpy import array,sqrt,vstack,ones,linspace
import pylab as pl
from pypwdg.core.boundary_data import zero_impedance

class SchwarzInterface(object):
    """Pass in mesh object and solution object and
    get values or derivs at points
    """
    def __init__(self, mesh, solution):
        self.mesh = mesh
        self.solution = solution
    
    def values(self, points, n=None):
        evalu = pce.Evaluator(self.mesh, self.solution.elttobasis, points)
        return evalu.evaluate(self.solution.x)        
    
    def derivs(self, points, n):
        evalu = pce.Evaluator(self.mesh, self.solution.elttobasis, points, n)
        return evalu.evaluate(self.solution.x)[1]
    

k = 20
direction=array([[1.0,1.0]])/sqrt(2)
g = pcb.PlaneWaves(direction, k)
impbd = pcbd.generic_boundary_data([-1j*k, 1], [-1j*k, 1], g)

bnddata1={7:zero_impedance(k),  #right
         8:impbd,
         9:impbd, #left
         10:impbd}

bounds1=array([[0,1],[0,1]],dtype='d')
bounds2=array([[1,2],[0,1]],dtype='d')
npoints=array([1000,1000])

mesh1 = pmm.gmshMesh('square.msh',dim=2)
mesh2 = pmm.gmshMesh('square2.msh', dim=2)
bases1 = pcb.planeWaveBases(mesh1,k,nplanewaves=15)
bases2 = pcb.planeWaveBases(mesh2,k,nplanewaves=15)

problem1 = ps.Problem(mesh1, k, 20, bnddata1)
solution1 = ps.Computation(problem1, bases1).solve()

#points = vstack([ones(1000), linspace(0, 1, 1000)]).transpose()

interface_data = pcbd.generic_boundary_data([-1j*k, 1],[-1j*k, 1], SchwarzInterface(mesh1, solution1))
bnddata2={7:impbd,  #right
         8:impbd,
         9:interface_data, #left
         10:impbd}

problem2 = ps.Problem(mesh2, k, 20, bnddata2)
solution2 = ps.Computation(problem2, bases2).solve()

solution1.writeSolution(bounds1,npoints,fname='firstdd.vti')
problem1.writeMesh(fname='firstdd.vtu',scalars=solution1.combinedError())
solution2.writeSolution(bounds2, npoints, fname='firstdd.vti')
problem1.writeMesh(fname='firstdd.vtu',scalars=solution2.combinedError())



