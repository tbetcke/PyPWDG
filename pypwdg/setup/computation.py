'''
Created on Apr 12, 2011

@author: joel
'''

import numpy
from pypwdg.core.vandermonde import LocalVandermondes
from pypwdg.core.physics import assemble
import pypwdg.core.bases as pcb

class Computation(object):
    """ Contains everything needed to solve a Problem  """
    def __init__(self, problem, elttobasis, usecache = True, dovols = False):
        self.problem = problem  
        self.elttobasis = elttobasis
        # Setup local Vandermondes        
        self.lv = LocalVandermondes(problem.mesh, elttobasis, problem.mqs, usecache=usecache)
        self.bndvs = []
        for data in problem.bnddata.values():
            bdyetob = pcb.constructBasis(problem.mesh, pcb.UniformBasisRule([data]))
            bndv = LocalVandermondes(problem.mesh, bdyetob, problem.mqs)        
            self.bndvs.append(bndv)
        stiffness, rhs = assemble(problem.mesh, problem.k, self.lv, self.bndvs, problem.mqs, self.elttobasis, problem.bnddata, problem.params, problem.emqs, dovols)
        self.stiffness = stiffness.tocsr()
        self.rhs = numpy.array(rhs.todense()).squeeze()
            
    def solve(self, solver="pardiso"):
        from pypwdg.setup.solution import Solution
        print "Solve linear system of equations"
        
        usepardiso = solver == "pardiso"
        useumfpack = solver == "umfpack"
        
        if not (usepardiso or useumfpack): raise Exception("Solver not known")
        
        if usepardiso:
            try:            
                from pymklpardiso.linsolve import solve
                (x, error) = solve(self.stiffness, self.rhs)
                if not error == 0: raise Exception("Pardiso Error")
            except ImportError:
                useumfpack = True
                
        if useumfpack:
            from scipy.sparse.linalg.dsolve.linsolve import spsolve as solve
            x = solve(self.stiffness, self.rhs)
        
        
        print "Relative residual: ", numpy.linalg.norm(self.stiffness * x - self.rhs) / numpy.linalg.norm(x)
        return Solution(self.problem, x, self.elttobasis, self.lv, self.bndvs)
                

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
            
        
        
    