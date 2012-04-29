'''
Created on Jul 27, 2010

@author: tbetcke
'''
import numpy as np
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
    
def gmshInfo(fname, dim, *args, **kwargs):
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
    return SimplicialMeshInfo(nodes, elements, elemIdentity, boundaries, dim, *args, **kwargs)

def gmshMesh(*args, **kwargs):
    meshinfo = gmshInfo(*args, **kwargs)
    topology = Topology(meshinfo)    
    meshview = DistributedMeshView(meshinfo, topology)
    return meshview

def compute_facedata(nodes, faces, nonfacevertex, dim):
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
    vertices = np.array([[nodes[fv] for fv in fvs] + [nodes[otherv]] for fvs, otherv in zip(faces, nonfacevertex)])
    # M picks out the first coord and the differences to the others
    M = np.bmat([[np.mat([[1]]), np.zeros((1,dim))], [np.ones((dim,1))*-1, np.eye(dim)]])
    # Apply the differencing matrix to each set of coordinates
    dirs = np.tensordot(vertices, M, ([1],[1]))
    # Ensure that the directions live in the last dimension
    directions = np.transpose(dirs, (0,2,1))
    
    normals=np.zeros((len(vertices),dim))
    
    if dim==1:
        normals[:,0]=directions[:,1,0]
        
    elif dim==2:
        # Put normal vectors 
        normals[:,0]=directions[:,1,1]
        normals[:,1]=-directions[:,1,0]
    else:
        normals[:,0]=directions[:,1,1]*directions[:,2,2]-directions[:,2,1]*directions[:,1,2]
        normals[:,1]=directions[:,1,2]*directions[:,2,0]-directions[:,1,0]*directions[:,2,2]
        normals[:,2]=directions[:,1,0]*directions[:,2,1]-directions[:,2,0]*directions[:,1,1]

    # this is 100x faster than applying np.linalg.norm to each entry
    dets = np.sqrt(np.sum(normals * normals, axis = 1))
            
    normals *= (-np.sign(np.sum(normals * directions[:,-1,:], axis = 1)) / dets ).reshape((-1,1))
    if dim==1: dets=np.ones(dets.shape,dtype=np.float64)

    return directions, normals, dets

def partition(elements, nnodes, nparts, dim):
    """ Return a partition of the elements of the mesh into nparts parts """ 
    allelts = np.arange(len(elements))
    if nparts ==1:
        return [allelts]
    else:
        """ Partition the mesh into nparts partitions """
        if dim==2:
            elemtype=1
        elif dim==3:
            elemtype=2
           
        (epart,npart,edgecut)=pymeshpart.mesh.part_mesh_dual(elements,nnodes,elemtype,nparts)
        return [allelts[epart == p] for p in range(nparts)]   

@ppd.immutable    
class SimplicialMeshInfo(object):
    def __init__(self, nodes, elements, elemIdentity, boundaries, dim):
        self.dim = dim
        self.elements = elements
        self.nodes = np.array(nodes)
        self.elemIdentity=elemIdentity
        self.nnodes=len(nodes)        
        self.nelements = len(elements)
        
        nev = self.dim+1        
        # The vertices associated with each face
        self.faces = np.array([tuple(e[0:i]+e[i+1:nev]) for e in self.elements for i in range(0,nev)])
        # The "opposite" vertex for each face    
        nonfacevertex = np.array([e[i] for e in self.elements for i in range(0,nev)])    

        self.nfaces = len(self.faces)
        self.ftoe = np.repeat(np.arange(self.nelements), self.dim+1)
        self.etof = np.arange(self.nfaces).reshape((-1, self.dim+1))
        self.elttofaces = ss.csr_matrix((np.ones(self.nfaces), np.concatenate(self.etof), np.cumsum([0] + map(len, self.etof))))        
        self.directions, self.normals, self.dets = compute_facedata(nodes, self.faces, nonfacevertex, dim)
        self.boundaries = boundaries
    
    def partition(self, nparts):
        return partition(self.elements, self.nnodes, npart, self.dim)

@ppd.immutable                
class Topology(object):
    def __init__(self, meshinfo):
        # the face to vertex map        
        ftov = ss.csr_matrix((np.ones(meshinfo.nfaces * meshinfo.dim), meshinfo.faces.ravel(), np.arange(0, meshinfo.nfaces+1)*meshinfo.dim), dtype=int)

        # determine the boundary faces        
        self.vtof = ftov.transpose().tocsr()
                
        self.entities = set() # Used for ray tracing
        self.faceentities = np.empty(meshinfo.nfaces, dtype=object)
        nb = np.ones(meshinfo.nfaces, dtype=int)
        for entityid, bnodes in meshinfo.boundaries:
            vs = set(bnodes)
            for v in vs:
                for f in self.vtof.getrow(v).indices:
                    if vs.issuperset(meshinfo.faces[f]): 
                        self.entities.add(entityid)            
                        self.faceentities[f] = entityid
                        nb[f] = 0
        
        print nb
        # anything assigned to a boundary should not be involved in connectivity (this allows for internal boundaries)
        nonboundary = ss.spdiags(nb, 0, meshinfo.nfaces,meshinfo.nfaces)
        nbftov = nonboundary * ftov 
                
        # now determine connectivity.  
        ftov2=(nbftov*nbftov.transpose()).tocsr() # Multiply to get connectivity.
        ftof = ss.csr_matrix((ftov2.data / meshinfo.dim, ftov2.indices, ftov2.indptr))  # ftov2 contains integer data, so dividing by dim means we're left with matching faces
        
        self.connectivity = ftof - nonboundary
        self.connectivity.eliminate_zeros()
        self.internal = self.connectivity **2
        self.boundary = ss.eye(meshinfo.nfaces, meshinfo.nfaces, dtype=int) - self.internal
        self.boundary.eliminate_zeros()
        bdyunassigned = (self.boundary * nonboundary).getnnz()        
        if bdyunassigned:
            print "Warning: %s non-internal faces not assigned to physical entities" %bdyunassigned
            print [self.faces[fid] for fid in self.boundary.diagonal().nonzero()[0] if self.faceentities[fid]==None]

class MeshView(object):
    def __init__(self, basicinfo, topology, partition = None):
        self.basicinfo = basicinfo
        self.topology = topology
        self.initialiseview(partition)
    
    def initialiseview(self, partition):
        self.partition = np.arange(self.nelements) if partition is None else partition 
        self.fs = self.basicinfo.etof[partition].ravel()
        fpindex = np.zeros((self.basicinfo.nfaces,), dtype=int)
        fpindex[self.fs] = 1
        nf = self.basicinfo.nfaces
        self.fp = ss.spdiags(fpindex, [0], nf, nf)
        
        self.cutfaces = (self.topology.internal - self.topology.connectivity) * fpindex
        cutelts = self.elttofaces * self.cutfaces
        self.neighbourelts = (cutelts <= -1).nonzero()[0]
        self.innerbdyelts = (cutelts >= 1).nonzero()[0]
        
    dim = property(lambda self: self.basicinfo.dim)
    nelements = property(lambda self: self.basicinfo.nelements)
    elements = property(lambda self: self.basicinfo.elements)
    nodes = property(lambda self: self.basicinfo.nodes)
    nfaces = property(lambda self: self.basicinfo.nfaces)
    ftoe = property(lambda self: self.basicinfo.ftoe)
    etof = property(lambda self: self.basicinfo.etof)
    elttofaces = property(lambda self: self.basicinfo.elttofaces)
    
    normals = property(lambda self: self.basicinfo.normals)
    directions = property(lambda self: self.basicinfo.directions)
    dets = property(lambda self: self.basicinfo.dets)
    
    facepartition = property(lambda self: self.fp)
    connectivity = property(lambda self: self.fp * self.topology.connectivity) 
    internal = property(lambda self: self.fp * self.topology.internal)
    boundary = property(lambda self: self.fp * self.topology.boundary)

    @property
    def entityfaces(self):
        return dict([(entity, self.fp * ss.spdiags((self.topology.faceentities == entity) * 1, [0], self.nfaces,self.nfaces)) for entity in self.topology.entities])

@ppd.distribute(lambda n: lambda basicinfo, topology: [((basicinfo, topology, partition, i),{}) for i, partition in enumerate(basicinfo.partitions(n))]) 
class DistributedMeshView(MeshView):
    pass    

