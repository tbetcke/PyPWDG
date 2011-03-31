import pypwdg.setup as ps
import pypwdg.core.bases as pcb
import pypwdg.mesh.mesh as pmm
import pypwdg.core.boundary_data as pcbd
import pypwdg.core.evaluation as pce
from numpy import array,sqrt
from pypwdg.core.boundary_data import zero_impedance, zero_dirichlet,\
    zero_neumann
    

class SchwarzInterface(object):
    """Pass in mesh object and solution object and
    get values or derivs at points
    """
    def __init__(self, mesh, solution):
        """Constructor for InterfaceBata, requires a Mesh object 
        and a Solution object"""
        self.mesh = mesh
        self.solution = solution
    
    def values(self, points, n=None):
        """Provides required values method for boundary data function"""
        evalu = pce.Evaluator(self.mesh, self.solution.elttobasis, points)
        return evalu.evaluate(self.solution.x).reshape(-1,1)
    
    def derivs(self, points, n):
        """Provides required derivs method for boundary data function"""
        evalu = pce.Evaluator(self.mesh, self.solution.elttobasis, points, n)
        return evalu.evaluate(self.solution.x)[1].reshape(-1,1)
    
#Wavenumber
k = 50

# #schwarz iterations
iterations = 2

#Problem is solve for incident wave coming in at direction wavenumber k

direction=array([[-1.0,3.0]])/sqrt(10)
g = pcb.FourierHankel(array((3., 2.)), [0], k)
impbd = pcbd.generic_boundary_data([-1j*k, 1], [-1j*k, 1], g)

bnddata={7:zero_neumann(),  #right
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

problem1 = ps.Problem(mesh1, k, 20, bnddata)
solution1 = ps.Computation(problem1, bases1).solve()


solution1.writeSolution(bounds1,npoints,fname='firstdd.vti')
#problem1.writeMesh(fname='firstdd.vtu',scalars=solution1.combinedError())

for i in range(iterations):
    print "Iteration", i
    interface_data = pcbd.generic_boundary_data([-1j*k, 1], [-1j*k, 1], SchwarzInterface(mesh1, solution1))
    bnddata={7:impbd,  #right
             8:impbd,
             9:interface_data, #left
             10:impbd}

    problem2 = ps.Problem(mesh2, k, 20, bnddata)
    solution2 = ps.Computation(problem2, bases2).solve()
    
    interface_data = pcbd.generic_boundary_data([-1j*k, 1], [-1j*k, 1], SchwarzInterface(mesh2, solution2))
    bnddata={7:interface_data,
             8:impbd,
             9:impbd,
             10:impbd}
    
    problem1 = ps.Problem(mesh1, k, 20, bnddata)
    solution1 = ps.Computation(problem1, bases1).solve()

solution1.writeSolution(bounds1,npoints,fname='firstddl.vti')
#problem1.writeMesh(fname='firstddl.vtu',scalars=solution1.combinedError())
solution2.writeSolution(bounds2, npoints, fname='firstddr.vti')
#problem1.writeMesh(fname='firstddr.vtu',scalars=solution2.combinedError())






