'''
Created on Jul 27, 2010

@author: tbetcke
'''
import numpy
import numpy as np
import numpy.ma as ma
import pymeshpart.mesh
import scipy.sparse as ss
import pypwdg.parallel.decorate as ppd
from pypwdg.mesh.gmsh_reader import gmsh_reader

class LineMesh(object):
    """Construct a mesh from a given interval
    
       INPUT:
       points    - An array [p0,p1,...,pN] defining the boundary points
                   of the physical identities. Have, p0<p1<...<pN. p0 and
                   pN are global boundary points. The interior points are
                   boundaries between physical entitites.
       nelems    - Array of length N defining the number of the elements in
                   each physical identity
       physIds   - Array of length N defining the physical Identities of the
                   elements
       bndIds    - Array of length 2 defining the boundary identitites
    """
    
    def __init__(self,points=[0,1],nelems=[20],physIds=[1],bndIds=[10,11]):
        self.nodes=np.zeros((0,1),dtype=np.float64)
        for i in range(len(points)-1):
            self.nodes=np.vstack((self.nodes,points[i]+(points[i+1]-points[i])*np.arange(nelems[i],dtype=np.float64).reshape(nelems[i],1)/nelems[i]))
        self.nodes=np.vstack((self.nodes,np.array([points[-1]])))
        self.elements=[[i,i+1] for i in range(len(self.nodes)-1)]    
        self.elemIdentity=[]
        for i in range(len(nelems)): self.elemIdentity+=[physIds[i]]*nelems[i]
        self.boundaries=[(bndIds[0],(0,)),(bndIds[1],(len(self.nodes)-1,))]
        
    def getMesh(self):
        return Mesh(self.nodes,self.elements,self.elemIdentity,self.boundaries,1)
    
    def refineElements(self,elemList):
        """Refine a list of elements
        
           INPUT
           elemList - List of element Ids to refine
        """
        
        newElements=[]
        newElemIdentity=[]
        newNodes=np.zeros((len(self.nodes)+len(elemList),1),dtype=np.float64)
        newNodes[:len(self.nodes)]=self.nodes
        offset=len(self.nodes)
        
        for id in range(len(self.elements)):
            if not id in elemList:
                newElements.append(self.elements[id])
                newElemIdentity.append(self.elemIdentity[id])
            else:
                nd=.5*(self.nodes[self.elements[id][0],0]+self.nodes[self.elements[id][1],0])
                newNodes[offset,0]=nd
                newElements.append([self.elements[id][0],offset])
                newElements.append([offset,self.elements[id][1]])
                newElemIdentity+=2*[self.elemIdentity[id]]
                offset+=1
        self.elements=newElements
        self.elemIdentity=newElemIdentity
        self.nodes=newNodes
        
    def refineAll(self):
        """Refine all elements"""
        
        self.refineElements(range(len(self.elements)))
        


def lineMesh(points=[0,1],nelems=[20],physIds=[1],bndIds=[10,11]):
    """Construct a mesh from a given interval
    
       INPUT:
       points    - An array [p0,p1,...,pN] defining the boundary points
                   of the physical identities. Have, p0<p1<...<pN. p0 and
                   pN are global boundary points. The interior points are
                   boundaries between physical entitites.
       nelems    - Array of length N defining the number of the elements in
                   each physical identity
       physIds   - Array of length N defining the physical Identities of the
                   elements
       bndIds    - Array of length 2 defining the boundary identitites
    """
    nodes=np.zeros((0,1),dtype=np.float64)
    for i in range(len(points)-1):
        nodes=np.vstack((nodes,points[i]+(points[i+1]-points[i])*np.arange(nelems[i],dtype=np.float64).reshape(nelems[i],1)/nelems[i]))
    nodes=np.vstack((nodes,np.array([points[-1]])))
    elements=[[i,i+1] for i in range(len(nodes)-1)]    
    elemIdentity=[]
    for i in range(len(nelems)): elemIdentity+=[physIds[i]]*nelems[i]
    boundaries=[(bndIds[0],(0,)),(bndIds[1],(len(nodes)-1,))]
    return Mesh(nodes, elements, elemIdentity, boundaries, 1)
    
def gmshMesh(fname, dim, *args, **kwargs):
    ''' Construct a Mesh from a gmsh dictionary '''
    
    if dim==2:
        gmsh_elem_key=2 # Key for triangle element in Gmsh 
        gmsh_face_key=1 # Key for line element in Gmsh
    elif dim==3:
        gmsh_elem_key=4 # Key for tetrahedral element in Gmsh
        gmsh_face_key=2 # Key for triangle element in Gmsh
    
    gmsh_dict=gmsh_reader(fname)
    
    # Pick out the coordinates of the vertices that we actually need
    nodes = gmsh_dict['nodes'][:,0:dim]
    
    els=filter(lambda e : e['type']==gmsh_elem_key, gmsh_dict['elements'].values())
    
    # This is the element->vertex map.  This establishes a canonical element ordering
    elements = map(lambda e: sorted(e['nodes']), els)
    
    # Create a map from the elements to their geometric Identities
    elemIdentity = map(lambda e: e['geomEntity'], els)
    
    # These are the physical entities in the mesh.
    boundaries = map(lambda f: (f['physEntity'], tuple(sorted(f['nodes']))), filter(lambda e : e['type']==gmsh_face_key, gmsh_dict['elements'].values()))
    return Mesh(nodes, elements, elemIdentity, boundaries, dim, *args, **kwargs)

class EtofInfo(object):
    def __init__(self, dim, nelts, nfaces):
        self.dim = dim
        self.nelements = nelts
        self.nfaces = nfaces
        self.ftoe = np.repeat(np.arange(nelts), dim+1)
        self.etof = np.arange(nfaces).reshape((-1,dim+1))
        self.elttofaces = ss.csr_matrix((np.ones(self.nfaces), np.concatenate(self.etof), np.cumsum([0] + map(len, self.etof))))

        
@ppd.immutable
class Mesh(EtofInfo):
    """Mesh - The structure of a simplicial mesh
       
       Usage:
       mesh=Mesh(nodes, elements, boundaries, dim)
           nodes: a sequence of (dim)-coordinates, giving the vertices of the mesh
           elements: a sequence of sorted (dim+1)-tuples, giving the vertices for each element in the mesh
           boundaries: a sequence of tuples, each of the form (id, nodes), identifying the physical objects in the mesh produced by the mesh generator
           dim: the dimension of the mesh (2 or 3, although other numbers probably work too)
           computeAll: should every partition compute properties for all faces, or just the relevant ones?
           
       Conceptually, a Mesh has three different types of property:
       1) Global properties are the same in each process - they are serialised normally.  
       2) Partition properties describe the partition of this Mesh held in the current process.  
       3) Masked properties are notionally global, but incomplete.  Only the data relevant
       to this process are held (the others are masked)   
       In the master process, partition and masked properties are not accessible
       
       Global properties:

            faces         - For each face-element pair, a tuple of vertices.  The ordering reflects the ordering of elements.  Each internal face appears twice
            nonfacevertex - For each face-element pair, the element vertex which does not appear on that face
            nfaces        - Number of faces = len(self.faces). Faces are counted twice if they are between two elements
            nnodes        - Number of nodes
            nodes         - List of nodes
            intfaces      - List of indices of faces in the interior
            bndfaces      - List of indices of faces on the boundary
            boundaries    - a sequence of tuples, each of the form (id, nodes), identifying the physical objects in the mesh
            elements      - List of elements
            nelements     - Number of elements
            dim           - Dimension of problem (dim=2,3)
            etof          - List of lists of faces for each element
            elemIdentity  - List of geometric identities of all elements

        Partition properties:
            fs            - The faces in this partition
            connectivity  - The connectivity of the faces in this partition (note that this will reference faces in other partitions) 
            internal      - Pull out the internal faces for this partition
            boundary      - Pull out the boundary faces for this partition
            entityfaces   - A dictionary of physical entity -> the corresponding faces in this partition            

        Masked properties:
            directions, normals, dets 
                          - see MeshPart.__compute_facedata
            
            
    The elements and faces of a mesh are given a canonical ordering by self.__faces and self.__elements   
    """
    
    def __init__(self,nodes, elements, elemIdentity, boundaries,dim, computeall = False):
        """ Initialize Mesh """
    
        self.elements = elements
        self.nodes = np.array(nodes)
        self.boundaries = boundaries
        self.elemIdentity=elemIdentity
        self.nnodes=len(nodes)
        
        nev = dim+1
        
        # The vertices associated with each face
        faces = numpy.array([tuple(e[0:i]+e[i+1:nev]) for e in elements for i in range(0,nev)])
        self.faces = faces
        # The "opposite" vertex for each face    
        self.nonfacevertex = numpy.array([e[i] for e in elements for i in range(0,nev)])    
        
        EtofInfo.__init__(self, dim, len(elements), len(faces))
                
        ftov = ss.csr_matrix((numpy.ones(self.nfaces * dim), faces.ravel(), np.arange(0, self.nfaces+1)*dim), dtype=int)
        ftov2=(ftov*ftov.transpose()).tocsr() # Multiply to get connectivity.
        ftof = ss.csr_matrix((ftov2.data / dim, ftov2.indices, ftov2.indptr))  # ftov2 contains integer data, so dividing by dim means we're left with matching faces
        
        self._connectivity = ftof - ss.eye(self.nfaces, self.nfaces, dtype=int)
        self._connectivity.eliminate_zeros()
        self._internal = self._connectivity **2
        self._boundary = ss.eye(self.nfaces, self.nfaces) - self._internal
        self._boundary.eliminate_zeros()
        
# Used for ray tracing:        
        self.entities = set()
        self.faceentities = np.empty(self.nfaces, dtype=object)
        
        self.vtof = ftov.transpose().tocsr()  
        for entityid, bnodes in boundaries:
            vs = set(bnodes)
            for v in vs:
                for f in self.vtof.getrow(v).indices:
                    if vs.issuperset(self.faces[f]): 
                        self.entities.add(entityid)            
                        self.faceentities[f] = entityid
                
        bdyunassigned = self._boundary.getnnz() - len(self.faceentities.nonzero()[0])        
        if bdyunassigned:
            print "Warning: %s non-internal faces not assigned to physical entities" %bdyunassigned
            print [self.faces[id] for id in self._boundary.diagonal().nonzero()[0] if self.faceentities[id]==None]

        self.directions, self.normals, self.dets = self.compute_facedata()
        self.meshpart = MeshPart(self)
    
    def partitions(self,nparts):
        """ Return a partition of the elements of the mesh into nparts parts """ 
        if nparts ==1:
            return [range(self.nelements)]
        else:
            """ Partition the mesh into nparts partitions """
            if self.dim==2:
                elemtype=1
            elif self.dim==3:
                elemtype=2
               
            (epart,npart,edgecut)=pymeshpart.mesh.part_mesh_dual(self.elements,self.nnodes,elemtype,nparts)
            return [np.arange(self.nelements)[epart == p] for p in range(nparts)]   
    
    def compute_facedata(self, relevantfaces = np.s_[:]):
        """ Compute directions, normals and determinants for the faces given by relevantfaces 
        
            The following arrays are computed:
            
            directions - three dimensional array, such that self.directions[ind] returns
                                a matrix:
                                    row 0 is the coordinate vector of the vertex v0. 
                                    rows 1:dim are the offsets to the other vertices of the face
                                    rows dim is the offset to the non-face vertex in the element 
            normals - Numpy array of dimension (self.nfaces,self.dim). self.normals[ind] contains the normal
                             direction of the face with index ind.
            dets    - Numpy array of dimension self.nfaces, containing the absolute value of the cross product of the
                             partial derivatives for the map from the unit triangle (or line in 2d) to the face        
        """
        # vertices is a 3-tensor.  For each (double sided) face, it contains a matrix of dim+1 coordinates.  The first dim of these
        # are on the face, the final one is the non-face vertex for the corresponding element
        vertices = numpy.array([[self.nodes[fv] for fv in fvs] + [self.nodes[otherv]] for fvs, otherv in zip(self.faces[relevantfaces], self.nonfacevertex[relevantfaces])])
        # M picks out the first coord and the differences to the others
        M = numpy.bmat([[numpy.mat([[1]]), numpy.zeros((1,self.dim))], [numpy.ones((self.dim,1))*-1, numpy.eye(self.dim)]])
        # Apply the differencing matrix to each set of coordinates
        dirs = numpy.tensordot(vertices, M, ([1],[1]))
        # Ensure that the directions live in the last dimension
        directions = numpy.transpose(dirs, (0,2,1))
        
        normals=numpy.zeros((len(vertices),self.dim))
        
        if self.dim==1:
            normals[:,0]=directions[:,1,0]
            
        elif self.dim==2:
            # Put normal vectors 
            normals[:,0]=directions[:,1,1]
            normals[:,1]=-directions[:,1,0]
        else:
            normals[:,0]=directions[:,1,1]*directions[:,2,2]-directions[:,2,1]*directions[:,1,2]
            normals[:,1]=directions[:,1,2]*directions[:,2,0]-directions[:,1,0]*directions[:,2,2]
            normals[:,2]=directions[:,1,0]*directions[:,2,1]-directions[:,2,0]*directions[:,1,1]
    
        # this is 100x faster than applying numpy.linalg.norm to each entry
        dets = numpy.sqrt(numpy.sum(normals * normals, axis = 1))
                
        normals *= (-numpy.sign(numpy.sum(normals * directions[:,-1,:], axis = 1)) / dets ).reshape((-1,1))
        if self.dim==1: dets=numpy.ones(dets.shape,dtype=numpy.float64)

        return directions, normals, dets
    
    es = property(lambda self: self.meshpart.es) 
    fs = property(lambda self: self.meshpart.fs)
    connectivity = property(lambda self: self.meshpart.fp * self._connectivity) 
    internal = property(lambda self: self.meshpart.fp * self._internal)
    boundary = property(lambda self: self.meshpart.fp * self._boundary)
    entityfaces = property(lambda self: self.meshpart.entityfaces)
#    directions = property(lambda self: self.meshpart.directions)
#    normals = property(lambda self: self.meshpart.normals)
#    dets = property(lambda self: self.meshpart.dets)
    partition = property(lambda self:self.meshpart.es)
    facepartition = property(lambda self: self.meshpart.fp)
    neighbourelts = property(lambda self:self.meshpart.neighbourelts)
    innerbdyelts = property(lambda self:self.meshpart.innerbdyelts)
    partitionidx = property(lambda self:self.meshpart.partidx)
    cutfaces = property(lambda self:self.meshpart.cutfaces)

@ppd.distribute(lambda n: lambda mesh: [((mesh, partition, i),{}) for i, partition in enumerate(mesh.partitions(n))]) 
class MeshPart(object):
    """ The Partition-specific data for a mesh"""
    def __init__(self, mesh, eltpartition=None, partidx = 0):
        if eltpartition == None: eltpartition = mesh.partitions(1)[0]
        self.partidx = partidx
        self.mesh = mesh
        self.es = eltpartition
        self.fs = mesh.etof[eltpartition].ravel()
        fpindex = np.zeros((mesh.nfaces,), dtype=int)
        fpindex[self.fs] = 1
        self.fp = ss.spdiags(fpindex, [0], mesh.nfaces, mesh.nfaces)
        
        self.cutfaces = (mesh._internal - mesh._connectivity) * fpindex
        cutelts = mesh.elttofaces * self.cutfaces
        self.neighbourelts = (cutelts == -1).nonzero()[0]
        self.innerbdyelts = (cutelts == 1).nonzero()[0]
        
        
        self.entityfaces = dict([(entity, self.fp * ss.spdiags((mesh.faceentities == entity) * 1, [0], mesh.nfaces, mesh.nfaces)) for entity in mesh.entities])
    
    

    
