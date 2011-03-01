'''
Created on 1 Nov 2010

@author: tbetcke
'''

__all__ = ["setup", "solve", "problem"]

import numpy
import pypwdg.core.bases as pcb
from pypwdg.utils.quadrature import trianglequadrature, legendrequadrature
from pypwdg.mesh.meshutils import MeshQuadratures
from pypwdg.core.vandermonde import LocalVandermondes
from pypwdg.core.physics import assemble
from pypwdg.core.evaluation import StructuredPointsEvaluator
from pypwdg.output.vtk_output import VTKStructuredPoints
from pypwdg.output.vtk_output import VTKGrid

import pypwdg.core.evaluation as pce

import pypwdg.parallel.main

problem = None
computation = None
usecache = True

def setup(mesh,k,nquadpoints,bnddata):
    """ Convenience method to create global Problem object"""
    global computation, problem
    computation = None
    problem = Problem(mesh,k,nquadpoints,bnddata)
    return problem

def computation(elttobasis):
    """ Convenience method to create global Computation object"""
    global computation, usecache
    computation = Computation(problem, elttobasis, usecache)

def solve(elttobasis = None, solver="pardiso"):
    """ Convenience method to solve the global Problem"""
    if elttobasis: computation(elttobasis)
    return computation.solve(solver)    

class Problem(object):
    """ The definition of a problem to be solved

       mesh        - A Mesh object
       k           - Wavenumber of the problem
       nquadpoints - Number of quadrature points
       nplanewaves - Number of Plane Waves in each element
                     In 3D the actual number of planewaves used is 6*nplanewaves^2
       bnddata     - Dictionary containing the boundary data
                     The dictionary takes the form bnddata[id]=bndobject,
                     where id is an identifier for the corresponding boundary and bndobject is an object defining
                     the boundary data (see pypwdg.core.boundary_data)
        
        Example:             
        problem=Problem(gmshMesh('myMesh.msh',dim=2),k=5,nquadpoints=10, elttobasis, bnddata={5: dirichlet(g), 6:zero_impedance(k)})
    """
    
    def __init__(self,mesh,k,nquadpoints,bnddata):
        self.mesh=mesh
        self.k=k
        self.bnddata=bnddata
        self.params=None
                
        # Set DG Parameters        
        self.setParams()
        
        # Set-up quadrature rules        
        if mesh.dim == 2:
            quad = legendrequadrature(nquadpoints)
        else:
            quad = trianglequadrature(nquadpoints)
        self.mqs = MeshQuadratures(self.mesh, quad)
                    
    def setParams(self,alpha=0.5,beta=0.5,delta=0.5):
        self.params={'alpha':alpha,'beta':beta,'delta':delta}        
        
    def writeMesh(self, fname='mesh.vtu', scalars=None):
        vtkgrid = VTKGrid(self.mesh, scalars)
        vtkgrid.write(fname)
        

class Computation(object):
    """ Contains everything needed to solve a Problem  """
    def __init__(self, problem, elttobasis, usecache = True):
        self.problem = problem  
        self.elttobasis = elttobasis
        # Setup local Vandermondes        
        self.lv = LocalVandermondes(problem.mesh, elttobasis, problem.mqs, usecache=usecache)
        self.bndvs = []
        for data in problem.bnddata.values():
            bndv = LocalVandermondes(problem.mesh, pcb.ElementToBases(problem.mesh).addUniformBasis(data), problem.mqs)        
            self.bndvs.append(bndv)
        stiffness, rhs = assemble(problem.mesh, problem.k, self.lv, self.bndvs, problem.mqs, self.elttobasis, problem.bnddata, problem.params)
        self.stiffness = stiffness.tocsr()
        self.rhs = numpy.array(rhs.todense()).squeeze()
            
    def solve(self, solver="pardiso"):
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
        print "Evaluate Solution and Write to File", fname
        
        bounds=numpy.array(bounds,dtype='d')
        filter=numpy.real if realdata else numpy.imag

        vtk_structure=VTKStructuredPoints(StructuredPointsEvaluator(self.problem.mesh, self.elttobasis, filter, self.x))
        vtk_structure.create_vtk_structured_points(bounds,npoints)
        vtk_structure.write_to_file(fname)
   
    def evalJumpErrors(self):
        print "Evaluate Jumps"
        (self.error_dirichlet2, self.error_neumann2, self.error_boundary2) = pce.EvalElementError3(self.problem.mesh, self.problem.mqs, self.lv, self.problem.bnddata, self.bndvs).evaluate(self.x)

    def combinedError(self):        
        if self.error_dirichlet2 is None: self.evalJumpErrors()
        error_combined2 = self.problem.k ** 2 * self.error_dirichlet2 + self.error_neumann2 + self.error_boundary2
        self.error_combined = numpy.sqrt(error_combined2)
        return self.error_combined
            
        
        
    
        
        
        
