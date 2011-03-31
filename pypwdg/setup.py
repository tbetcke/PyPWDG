'''
Created on 1 Nov 2010

@author: tbetcke
'''

__all__ = ["setup", "solve", "problem"]

import numpy as np
import pypwdg.core.bases as pcb
from pypwdg.utils.quadrature import trianglequadrature, legendrequadrature
from pypwdg.mesh.meshutils import MeshQuadratures
from pypwdg.core.vandermonde import LocalVandermondes
from pypwdg.core.physics import assemble
from pypwdg.core.evaluation import StructuredPointsEvaluator
from pypwdg.output.vtk_output import VTKStructuredPoints
from pypwdg.output.vtk_output import VTKGrid
from pypwdg.utils.rich import write_out

from pypwdg.utils.timing import print_timing

import pypwdg.core.evaluation as pce

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
            bdyetob = pcb.constructBasis(problem.mesh, pcb.UniformBases([data]))
            bndv = LocalVandermondes(problem.mesh, bdyetob, problem.mqs)        
            self.bndvs.append(bndv)
        stiffness, rhs = assemble(problem.mesh, problem.k, self.lv, self.bndvs, problem.mqs, self.elttobasis, problem.bnddata, problem.params)
        self.stiffness = stiffness.tocsr()
        self.rhs = np.array(rhs.todense()).squeeze()
    
    @print_timing        
    def solve(self, solver="pardiso", precond=None, part=None):
        print "Solve linear system of equations"
        
        usepardiso = solver == "pardiso"
        useumfpack = solver == "umfpack"
        usecg      = solver == "cg"
        usegmres   = solver == "gmres"
        usebicgstab = solver == "bicgstab"
        
        if not (usepardiso or useumfpack or usecg or usegmres or usebicgstab): raise Exception("Solver not known")
        
        if usepardiso:
            try:            
                from pymklpardiso.linsolve import solve
                (x, error) = solve(self.stiffness, self.rhs)
                if not error == 0: raise Exception("Pardiso Error")
            except ImportError:
                useumfpack = True
        
        if usecg:
            from scipy.sparse.linalg import cg
            (x, error) = cg(self.stiffness, self.rhs)
            print x   
            
        if usebicgstab:
            from scipy.sparse.linalg import bicgstab
            (x, error) = bicgstab(self.stiffness, self.rhs)
            
        if useumfpack:
            from scipy.sparse.linalg.dsolve.linsolve import spsolve
            x = spsolve(self.stiffness, self.rhs)
        
        if usegmres:
            print "Using gmres iterative solver with pre:", precond
            residues = []
            def callback(x):
                residues.append(x)
            
            from scipy.sparse.linalg.isolve import gmres
            from pypwdg.utils.preconditioning import block_diagonal, diagonal
            M = None
            if precond == 'diag':
                M = diagonal(self.stiffness)
            if precond == 'block_diag':
                if part == 'elms':
                    partitions = [np.array([i]) for i in range(self.problem.mesh.nelements)]
                elif type(part) == type(1):
                    partitions = self.problem.mesh.partitions(part)
                else:
                    print "Partition number not understood - defaulting to 2"
                    partitions = self.problem.mesh.partitions(2)
                idxs = [np.concatenate([np.arange(self.elttobasis.getIndices()[e], \
                            self.elttobasis.getIndices()[e] + self.elttobasis.getSizes()[e]) \
                            for e in partition]) for partition in partitions]
                M = block_diagonal(self.stiffness, idxs)
            (x, error) = gmres(self.stiffness, self.rhs, tol=1e-10, restart=2000, M=M, callback=callback)
            print "Gmres error code: ", error
            print "Residue:", residues[-1]
            write_out(residues, "./residues.dat")

        if usebicgstab:
            print "Using bicgstab iterative solver with pre:", precond
            from scipy.sparse.linalg.isolve import bicgstab
            from pypwdg.utils.preconditioning import block_diagonal, diagonal
            M = None
            if precond == 'diag':
                M = diagonal(self.stiffness)
            if precond == 'block_diag':
                partitions = self.problem.mesh.partitions(3)
                idxs = [np.concatenate([np.arange(self.elttobasis.getIndices()[e], self.elttobasis.getIndices()[e] + self.elttobasis.getSizes()[e]) for e in partition]) for partition in partitions]
                M = block_diagonal(self.stiffness, idxs)
            x, error = bicgstab(self.stiffness, self.rhs, M=M)
            print "Bicgstab error code: ", error

            
        print "Relative residual: ", np.linalg.norm(self.stiffness * x - self.rhs) / np.linalg.norm(x)
        return Solution(self.problem, x, self.elttobasis, self.lv, self.bndvs)
                

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
        print "Evaluate Solution and Write to File", fname
        
        bounds=np.array(bounds,dtype='d')
        filter=np.real if realdata else np.imag

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
        self.error_combined = np.sqrt(error_combined2)
        return self.error_combined
            
        
        
    
        
        
        
