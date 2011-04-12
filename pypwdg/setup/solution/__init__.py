
import numpy
from pypwdg.core.vandermonde import LocalVandermondes
from pypwdg.core.evaluation import StructuredPointsEvaluator
from pypwdg.output.vtk_output import VTKStructuredPoints
import pypwdg.core.evaluation as pce

def noop(x):
    return x


class Solution(object):
    """ The solution to a Problem """
    def __init__(self, problem, x, elttobasis, lvs, bndvs):  
        self.problem = problem
        self.x = x
        self.elttobasis = elttobasis
        self.lv = lvs
        self.bndvs = bndvs
        self.error_dirichlet2=None
        self.error_neumann2=None
        self.error_boundary2=None
        self.error_combined=None
        
              
    def writeSolution(self, bounds, npoints, realdata=True, fname='solution.vti'):
        print "Evaluate Solution and Write to File"
        
        bounds=numpy.array(bounds,dtype='d')
        filter=numpy.real if realdata else numpy.imag

        vtk_structure=VTKStructuredPoints(StructuredPointsEvaluator(self.problem.mesh, self.elttobasis, filter, self.x))
        vtk_structure.create_vtk_structured_points(bounds,npoints)
        vtk_structure.write_to_file(fname)
   
    def evaluate(self, structuredpoints):
        spe = StructuredPointsEvaluator(self.problem.mesh, self.elttobasis, noop, self.x)
        vals, count = spe.evaluate(structuredpoints)
        count[count==0] = 1
        return vals / count
   
    def evalJumpErrors(self):
        print "Evaluate Jumps"
        (self.error_dirichlet2, self.error_neumann2, self.error_boundary2) = pce.EvalElementError3(self.problem.mesh, self.problem.mqs, self.lv, self.problem.bnddata, self.bndvs).evaluate(self.x)

    def combinedError(self):        
        if self.error_dirichlet2 is None: self.evalJumpErrors()
        error_combined2 = self.problem.k ** 2 * self.error_dirichlet2 + self.error_neumann2 + self.error_boundary2
        self.error_combined = numpy.sqrt(error_combined2)
        return self.error_combined
            
        