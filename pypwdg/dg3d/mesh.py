'''
Created on May 26, 2010
A test comment
@author: joel
'''
from meshpy.tet import MeshInfo, build, Options
import math
import numpy
from numpy import  mat, identity, ones,hstack,vstack,zeros,dot,asmatrix
from pypwdg.dg3d.utils import *

class Mesh(object):
    """ Encapsulates properties of the mesh 
    
        Attributes:
        points - vertices as list of tuples
        dim - dimension of manifold (3 for the moment)
        ftovs - list of vertices for each (one-sided) face
        etof - list of (two-sided) faces for each element
        boundaryfaces - list of boundary faces (one-sided)
        normals - list of normals for each (two-sided) face
        dsfs - list of one-sided faces corresponding to each double-sided face
        connectivity - sparse connectivity matrix for double-sided faces
        boundary - sparse matrix that picks out boundary faces
        average - averages across faces
        jump - jumps across faces
        sfaverage - two-sided face to one-sided face averaging (and identity on boundary)
    """
    def __init__(self, mesh):
        self._inittetmesh(mesh)
        self._initmesh()
    
    def _inittetmesh(self, mesh):
        """Extract the important information from the tetmesh description"""
        
        #Although we keep _mesh as an attribute, this is the only place it should be referred to
        self._mesh = mesh
        self.points = numpy.array(mesh.points)
        # Solver isn't fully dimension-aware yet, but a few bits are (e.g. NormalM)
        self.dim = self.points.shape[1]
        
        # ftovs contains the vertices for each of the (one-sided) faces in the mesh
        # The ordering, and hence face number, is implied by the lexicographical ordering of the vertices 
        self.ftovs = sorted(set([tuple(sorted(vs[0:i]+vs[i+1:4])) for vs in mesh.elements for i in range(0,4)]))
        # Now determine the faces for each vertex
        vtof = {}
        for x,vs in enumerate(self.ftovs):
            for v in vs:
                if v not in vtof: vtof[v] = []
                vtof[v].append(x)
        # Now determine the faces for each element.  There's probably a sexier way to do this.
        self.etof = []         
        for vs in mesh.elements:
            fs = []
            for v in vs:
                for f in vtof[v]:
                    if set(self.ftovs[f]).issubset(set(vs)) and f not in set(fs): 
                        fs.append(f)
                    # could break here when len(fs) = 4            
            self.etof.append(fs)
        # Extract the boundary faces.  This could be inferred from connectivity, but the mesh generator
        # supports boundary labelling.  We're not using this yet, but when we do, it's implementation-specific
        mf = [set(vs) for vs in mesh.faces]
        self.boundaryfaces = [f for f,vs in enumerate(self.ftovs) if set(vs) in mf]
        self.normals = [[self.outwardNormal(mesh.elements[e],self.ftovs[f]) for f in fs] for e,fs in enumerate(self.etof)  ]
        
    
    def _initmesh(self):
        """Implementation-independent mesh initialisation"""
        from scipy import sparse
        import numpy.matlib
        
        # Connectivity matrix
        self.dsfs = numpy.hstack(self.etof) # double-sided faces to faces
        nfs = len(self.dsfs)
        dftof = sparse.csr_matrix((numpy.ones(nfs), self.dsfs, range(nfs+1)))
        self.dftof = dftof
        connectivity = dftof * dftof.transpose()
        connectivity.setdiag(zeros(nfs))
        connectivity.eliminate_zeros()      
        
        # pick out the internal faces
        internal = sparse.csr_matrix((connectivity.data, sorted(list(connectivity.indices)), connectivity.indptr),shape=(nfs,nfs))
        # pick out the boundary faces
        self.boundary = sparse.spdiags(1 - internal.diagonal(),0,nfs,nfs).tocsr() 
    
        self.average = (connectivity + internal)/2
        self.jump = internal - connectivity
        
        # single face average 
        colsums = numpy.matlib.ones((1,nfs)) * dftof   # 2 for an interior face, 1 for a boundary face
        nf = dftof.get_shape()[1]
        self.sfaverage = (dftof * sparse.spdiags(1.0/colsums.A.squeeze(),0,nf,nf)).transpose().tocsr()
                   
        
    def outwardNormal(self, e, f):
        """ Compute the outward normal for the face f on element e"""
        
        # Would be faster to do all the faces at once, but probably doesn't matter too much
        n = normalM(self.points[numpy.array(f)])
        otherpoint = list(set(e) - set(f))[0]
        return n * numpy.sign(dot(numpy.array(self.points[f[0]]-self.points[otherpoint]),n))
            
    def facePoints(self, refpoints):
        """ Given some points on the reference triangle, map them to the faces. 
    
        returns a list of lists of 3-tuples."""
        D = mat([[-1,1,0],[-1,0,1]]) # difference matrix. D [v1,v2,v3]^t = [v2-v1, v3-v1]^t
        O = ones((refpoints.shape[0],1)) # offset matrix - used to pick out lots of copies of v1
        return [refpoints * D * P + O * P[0] for P in [asmatrix(self.points[numpy.array(vs)]) for vs in self.ftovs] ] 
    
            
    @print_timing
    def localvandermondes(self, refpoints, fd, fn):
        blockvd = self.values(refpoints, fd)
        blockvn = self.values(refpoints, fn)
        return blockvd, blockvn 
    
    def vandermonde(self, refpoints, fd,fn):
        """Compute the Vandermonde matrix """
        from scipy import sparse
#        facepoints = self.facePoints(refpoints)
        blockvd, blockvn = self.localvandermondes(refpoints, fd, fn)
        indices = numpy.repeat(range(len(self.etof)),self.dim+1) 
        indptr = range(len(indices)+1)        
        bvd = sparse.bsr_matrix((numpy.array(blockvd), indices, indptr))
        bvn = sparse.bsr_matrix((numpy.array(blockvn), indices, indptr))

        return bvd,bvn
    
    def values(self, refpoints, g):
        """ return the values of g at refpoints on each element in the mesh"""
        facepoints = self.facePoints(refpoints)
        return [g(facepoints[f],n) for (fs, ns) in zip(self.etof, self.normals) for (f,n) in zip(fs, ns)  ]
        
     
    def faceareas(self):
        """ returns a list of areas of all the faces in the mesh"""
        import numpy.matlib
        # de Gua's theorem; the area of a simplex is 1/n! times the area of a parallepiped;
        # the area of the parallepipied is the determinant of the points plus a column of ones
        # (check this by performing row ops to get the differences, which doesn't affect the det)
        
        # if this seems painful, bear in mind that I could have put it all in one list comprehension
        fa = []
        dim = self.dim
        for f in hstack(self.etof):
            vs = self.points[numpy.array(self.ftovs[f])]
            det2s = [numpy.linalg.det(hstack((vs[:,range(i)],numpy.matlib.ones((dim,1)),vs[:,range(i+1,dim)])))**2 for i in range(dim)]
            fa.append(math.sqrt(sum(det2s))/math.factorial(dim-1))  
        return fa        

 
                    
def tetmesh():
    """ Create a mesh for the reference tet"""
    mesh_info = MeshInfo()
    mesh_info.set_points([(0,0,0), (1,0,0), (0,1,0),(0,0,1)])
    mesh_info.set_facets([
        [0,1,2],
        [0,1,3],
        [0,2,3],
        [1,2,3]])

    opts = Options('pq')
    return Mesh(build(mesh_info, opts))    

                    
def tetmesh2():
    """ Create a mesh for a translated reference tet"""
    mesh_info = MeshInfo()
#    mesh_info.set_points([(0,0,0), (0,0,1), (0,-1,0),(1,0,0)])
    mesh_info.set_points([(1,1,1), (1,1,2), (1,2,2),(2,1,2)])
#    mesh_info.set_points([(0,0,0), (1,0,-1), (0,2,0),(0,0,3)])
#    mesh_info.set_points([(0,0,0), (1,0,0), (0,2,0),(0,0,1)])
    mesh_info.set_facets([
        [0,1,2],
        [0,1,3],
        [0,2,3],
        [1,2,3]])

    opts = Options('pq')
    return Mesh(build(mesh_info, opts))    

    
def cubemesh(maxvol = 1):
    """ Create a mesh for the unit cube"""
    mesh_info = MeshInfo()
    mesh_info.set_points([
        (0,0,0), (1,0,0), (1,1,0), (0,1,0),
        (0,0,1), (1,0,1), (1,1,1), (0,1,1)
        ])
    mesh_info.set_facets([
        [0,1,2,3],
        [4,5,6,7],
        [0,4,5,1],
        [1,5,6,2],
        [2,6,7,3],
        [3,7,4,0],
        ])

    opts = Options('pq')#, verbose=True)
#    opts = Options('pqa0.02', verbose=True)
    return Mesh(build(mesh_info, opts, max_volume=maxvol))

      
def printMesh(meshinfo):
    print "Mesh Points:"
    for i,p in enumerate(meshinfo.points): print i,p
    print "Elements"
    for i,t in enumerate(meshinfo.elements): print i,t
    print "Faces:"
    for i,f in enumerate(meshinfo.faces): print i,f
    #print "Face markers:"
    #for i,f in enumerate(mesh.face_markers): print i,f

def displayMesh(mesh):
    from numpy import array
    from enthought.mayavi.mlab import triangular_mesh  
    p = array(mesh._mesh.points)
    f = array(mesh.ftovs)
#    triangular_mesh(p[:,0],p[:,1],p[:,2],f,opacity=0.3)
    triangular_mesh(p[:,0],p[:,1],p[:,2],f,representation="wireframe", opacity=0.8)
    triangular_mesh(p[:,0],p[:,1],p[:,2],f,representation="surface", opacity=0.1)

if __name__ == "__main__":
    cube=cubemesh()



