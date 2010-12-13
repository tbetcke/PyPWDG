'''
Created on 1 Nov 2010

@author: tbetcke
'''

import numpy
from pypwdg.core.bases import ElementToBases
from pypwdg.utils.quadrature import trianglequadrature, legendrequadrature
from pypwdg.mesh.meshutils import MeshQuadratures
from pypwdg.core.vandermonde import LocalVandermondes
from pypwdg.core.physics import assemble
from pypwdg.core.evaluation import StructuredPointsEvaluator, EvalElementError
from pypwdg.output.vtk_output import VTKStructuredPoints
from pypwdg.output.vtk_output import VTKGrid

import pypwdg.parallel.main

def setup(mesh,k,nquadpoints,elttobasis,bnddata,usecache=False):
    """Returns a 'computation' object that contains everything necessary for a PWDG computation.
    
       INPUT Parameters:
       mesh        - A Mesh object
       k           - Wavenumber of the problem
       nquadpoints - Number of quadrature points
       nplanewaves - Number of Plane Waves in each element
                     In 3D the actual number of planewaves used is 6*nplanewaves^2
       bnddata     - Dictionary containing the boundary data
                     The dictionary takes the form bnddata[id]=bndobject,
                     where id is an identifier for the corresponding boundary and bndobject is an object defining
                     the boundary data (see pypwdg.core.boundary_data)
       
       
       An example call may take the form
       
       comp=setup(gmshMesh('myMesh.msh',dim=2),k=5,nquadpoints=10, elttobasis, bnddata={5: dirichlet(g), 6:zero_impedance(k)})
    """
    
    return computation(mesh,k,nquadpoints,elttobasis,bnddata,usecache)

class computation(object):
    
    def __init__(self,mesh,k,nquadpoints,elttobasis,bnddata,usecache):
        self.mesh=mesh
        self.k=k
        self.nquadpoints=nquadpoints
        self.bnddata=bnddata
        self.params=None
        self.usecache=usecache
        self.assembledmatrix=None
        self.rhs=None
        self.error_dirichlet=None
        self.error_neumann=None
        self.error_boundary=None
        self.error_combined=None
        self.x=None
                
        # Set DG Parameters
        
        self.setParams()
        
        # Setup quadrature rules
        
        if mesh.dim == 2:
            self.quad = legendrequadrature(self.nquadpoints)
        else:
            self.quad = trianglequadrature(self.nquadpoints)
        self.mqs = MeshQuadratures(self.mesh, self.quad)
        
        # Setup basis functions
        self.elttobasis = elttobasis
        
        # Setup local Vandermondes
        
        self.lv = LocalVandermondes(self.mesh, self.elttobasis, self.mqs, usecache=self.usecache)
        self.bndvs = []
        for data in self.bnddata.values():
            bndv = LocalVandermondes(self.mesh, ElementToBases(self.mesh).addUniformBasis(data), self.mqs)        
            self.bndvs.append(bndv)
            
    def setParams(self,alpha=0.5,beta=0.5,delta=0.5):
        self.params={'alpha':alpha,'beta':beta,'delta':delta}
            
    def assemble(self):
        print "Assembling system"
        self.assembledmatrix, self.rhs = assemble(self.mesh, self.k, self.lv, self.bndvs, self.mqs, self.elttobasis, self.bnddata, self.params)
    
    def solve(self, solver="pardiso"):

        if self.assembledmatrix is None: self.assemble()

        print "Solve linear system of equations"

        self.assembledmatrix = self.assembledmatrix.tocsr()
        self.rhs = numpy.array(self.rhs.todense()).squeeze()
        
        usepardiso = solver == "pardiso"
        useumfpack = solver == "umfpack"
        
        if not (usepardiso or useumfpack): raise Exception("Solver not known")
        
        if usepardiso:
            try:            
                from pymklpardiso.linsolve import solve
                (self.x, error) = solve(self.assembledmatrix, self.rhs)
                if not error == 0: raise Exception("Pardiso Error")
            except ImportError:
                useumfpack = True
                
        if useumfpack:
            from scipy.sparse.linalg.dsolve.linsolve import spsolve as solve
            self.x = solve(self.assembledmatrix, self.rhs)
        
        
        print "Relative residual: ", numpy.linalg.norm(self.assembledmatrix * self.x - self.rhs) / numpy.linalg.norm(self.x)
        
        
    def writeSolution(self, bounds, npoints, realdata=True, fname='solution.vti'):
        
        if self.x is None: self.solve()
        
        print "Evaluate Solution and Write to File"
        
        bounds=numpy.array(bounds,dtype='d')
        filter=numpy.real if realdata else numpy.imag

        vtk_structure=VTKStructuredPoints(StructuredPointsEvaluator(self.mesh, self.elttobasis, filter, self.x))
        vtk_structure.create_vtk_structured_points(bounds,npoints)
        vtk_structure.write_to_file(fname)
        
    def writeMesh(self, fname='mesh.vtu', scalars=None):
        vtkgrid = VTKGrid(self.mesh, scalars)
        vtkgrid.write(fname)
   
    def evalJumpErrors(self):
        
        if self.x is None: self.solve()
        
        print "Evaluate Jumps"
        EvalError = EvalElementError(self.mesh, self.elttobasis, self.quad, self.bnddata, self.lv, self.bndvs)
        (self.error_dirichlet, self.error_neumann, self.error_boundary) = EvalError.evaluate(self.x)
        
    def combinedError(self):
        
        if self.error_dirichlet is None: self.evalJumpErrors()
        self.error_combined = self.k ** 2 * self.error_dirichlet ** 2 + self.error_neumann ** 2 + self.error_boundary ** 2
        self.error_combined = numpy.sqrt(self.error_combined)
        return self.error_combined
            
        
        
    
        
        
        
