'''
Created on Apr 12, 2011

@author: joel
'''
from pypwdg.utils.quadrature import trianglequadrature, legendrequadrature, tetquadrature
from pypwdg.mesh.meshutils import MeshQuadratures, MeshElementQuadratures
import pypwdg.core.bases as pcb
from pypwdg.output.vtk_output import VTKGrid
import pypwdg.core.bases.variable as pcbv

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
            fquad = legendrequadrature(nquadpoints)
            equad = trianglequadrature(nquadpoints)
        else:
            fquad = trianglequadrature(nquadpoints)
            equad = tetquadrature(nquadpoints)
        self.mqs = MeshQuadratures(self.mesh, fquad)
        self.emqs = MeshElementQuadratures(self.mesh, equad)
                    
    def setParams(self,alpha=0.5,beta=0.5,delta=0.5):
        self.params={'alpha':alpha,'beta':beta,'delta':delta}
    
    def constructBasis(self, basisrule):
        elementinfo = pcb.ElementInfo(self.mesh, self.k)
        return pcb.constructBasis(self.mesh, basisrule, elementinfo)  
        
    def writeMesh(self, fname='mesh.vtu', scalars=None):
        vtkgrid = VTKGrid(self.mesh, scalars)
        vtkgrid.write(fname)

class VariableNProblem(Problem):
    def __init__(self, mesh,k,nquadpoints,bnddata, entitytoN = None, pointtoN = None):
        Problem.__init__(self, mesh,k,nquadpoints,bnddata)
        if entitytoN is not None:
            self.elementinfo = pcbv.EntityNElementInfo(mesh, k, entitytoN)
        