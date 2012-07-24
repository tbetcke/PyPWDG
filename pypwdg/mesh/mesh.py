'''
Created on Jul 27, 2010

@author: tbetcke
'''
import numpy as np
import scipy.sparse as ss
import pypwdg.parallel.decorate as ppd
from pypwdg.mesh.gmsh_reader import gmsh_reader
import logging
log = logging.getLogger(__name__)

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
    partition = DistributedPartition(meshinfo, topology)    
    meshview = MeshView(meshinfo, topology, partition)
    return meshview

def compute_facedata(nodes, faces, nonfacevertex, dim, vdim):
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
    normals=np.zeros((len(vertices),vdim))
    
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
   # if dim==1: dets=np.ones(dets.shape,dtype=np.float64)   ??????????????????

    return directions, normals, dets

def partition(elements, nnodes, nparts, dim):
    """ Return a partition of the elements of the mesh into nparts parts """ 
    allelts = np.arange(len(elements))
    if nparts ==1:
        return [allelts]
    else:
        import pymeshpart.mesh
        """ Partition the mesh into nparts partitions """
        if dim==2:
            elemtype=1
        elif dim==3:
            elemtype=2
           
        (epart,npart,edgecut)=pymeshpart.mesh.part_mesh_dual(elements,nnodes,elemtype,nparts)
        return [allelts[epart == p] for p in range(nparts)]   

@ppd.immutable    
class SimplicialMeshInfo(object):
    ''' Defines a mesh with simplicial elements.
    
        Parameters:
            nodes: list of nodes
            elements: list of dim+1-tuples
            elemIdentity: confusingly named parameter that gives the geometric entity that each element belongs to
            boundaries: list of (tag, [nodes]) tuples that indicates that the face defined by [nodes] belongs to boundary 'tag'
            dim: dimension of the elements
            vdim: dimension of the space in which the mesh is embedded (e.g. a surface mesh would have dim=2 and vdim=3)
    '''             
    def __init__(self, nodes, elements, elemIdentity, boundaries, dim, vdim = None):
        self.dim = dim
        vdim = dim if vdim is None else vdim
        self.elements = map(tuple, elements)
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
        self.directions, self.normals, self.dets = compute_facedata(nodes, self.faces, nonfacevertex, dim, vdim)
        self.boundaries = boundaries
    
    def partition(self, nparts):
        return partition(self.elements, self.nnodes, nparts, self.dim)

@ppd.immutable                
class Topology(object):
    ''' Calculate and store the topological information for a mesh''' 
    def __init__(self, meshinfo):
        # the face to vertex map        
        ftov = ss.csr_matrix((np.ones(meshinfo.nfaces * meshinfo.dim), meshinfo.faces.ravel(), np.arange(0, meshinfo.nfaces+1)*meshinfo.dim), dtype=int)

        # determine the boundary faces        
        self.vtof = ftov.transpose().tocsr()
                
        self.entities = set() # Used for ray tracing
        self.faceentities = np.empty(meshinfo.nfaces, dtype=object)
        nb = np.ones(meshinfo.nfaces, dtype=int)
        
        # this allows you to do something quite dangerous with the boundaries.  It's best to define boundaries in terms of faces (not larger collections of vertices)
        for entityid, bnodes in meshinfo.boundaries:
            vs = set(bnodes)
            for v in vs:
                for f in self.vtof.getrow(v).indices:
                    if vs.issuperset(meshinfo.faces[f]): 
                        self.entities.add(entityid)            
                        self.faceentities[f] = entityid
                        nb[f] = 0
        
#        print nb
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
            log.warn("Warning: %s non-internal faces not assigned to physical entities" %bdyunassigned)
            log.debug([meshinfo.faces[fid] for fid in self.boundary.diagonal().nonzero()[0] if self.faceentities[fid]==None])


class Partition(object):
    ''' A partition of a mesh'''
    def __init__(self, basicinfo, topology, partition=None, partidx=0):
        self.partition = np.arange(basicinfo.nelements) if partition is None else partition 
        self.partidx = partidx
        self.fs = basicinfo.etof[partition].ravel()
        fpindicator = np.zeros((basicinfo.nfaces,), dtype=int)
        fpindicator[self.fs] = 1
        nf = basicinfo.nfaces
        self.fp = ss.spdiags(fpindicator, [0], nf, nf)
        
        self.cutfaces = (topology.internal - topology.connectivity) * fpindicator
        cutelts = basicinfo.elttofaces * self.cutfaces
        self.neighbourelts = (cutelts <= -1).nonzero()[0]
        self.innerbdyelts = (cutelts >= 1).nonzero()[0]
#        print "cut faces", (self.cutfaces==1).nonzero()[0], sum(self.cutfaces==1)
#        print 'Neighbour elements', self.neighbourelts
    

@ppd.distribute(lambda n: lambda basicinfo, topology: [((basicinfo, topology, partition, i),{}) for i, partition in enumerate(basicinfo.partition(n))]) 
class DistributedPartition(Partition):
    ''' A helper class that creates one mesh partition per worker process'''
    pass    

@ppd.distribute(lambda n: lambda basicinfo, topology, partitions: [((basicinfo, topology, partition, i),{}) for i, partition in enumerate(partitions(n))]) 
class BespokePartition(Partition):
    ''' A helper class that creates one mesh partition per worker process'''
    pass    

class MeshView(object):
    ''' Pulls all the mesh information together.  Provides a partition-specific view onto bits of the topology
    
        todo: document all the properties.
    '''
    def __init__(self, basicinfo, topology = None, partition=None):
        self.basicinfo = basicinfo
        self.topology = Topology(basicinfo) if topology is None else topology
        self.part = Partition(basicinfo, self.topology) if partition is None else partition
        
        
    dim = property(lambda self: self.basicinfo.dim)
    nelements = property(lambda self: self.basicinfo.nelements)
    elements = property(lambda self: self.basicinfo.elements)
    nodes = property(lambda self: self.basicinfo.nodes)
    nnodes = property(lambda self: self.basicinfo.nnodes)
    faces = property(lambda self: self.basicinfo.faces)
    nfaces = property(lambda self: self.basicinfo.nfaces)
    ftoe = property(lambda self: self.basicinfo.ftoe)
    etof = property(lambda self: self.basicinfo.etof)
    elttofaces = property(lambda self: self.basicinfo.elttofaces)
    
    normals = property(lambda self: self.basicinfo.normals)
    directions = property(lambda self: self.basicinfo.directions)
    dets = property(lambda self: self.basicinfo.dets)
    
    partition = property(lambda self: self.part.partition)
    cutfaces = property(lambda self: self.part.cutfaces)
    neighbourelts = property(lambda self: self.part.neighbourelts)
    innerbdyelts = property(lambda self: self.part.innerbdyelts)
    facepartition = property(lambda self: self.part.fp)
    connectivity = property(lambda self: self.part.fp * self.topology.connectivity) 
    internal = property(lambda self: self.part.fp * self.topology.internal)
    boundary = property(lambda self: self.part.fp * self.topology.boundary)

    @property
    def entityfaces(self):
        return dict([(entity, self.part.fp * ss.spdiags((self.topology.faceentities == entity) * 1, [0], self.nfaces,self.nfaces)) for entity in self.topology.entities])

def meshFromInfo(meshinfo):
    topology = Topology(meshinfo)
    partition = DistributedPartition(meshinfo, topology)
    return MeshView(meshinfo, topology, partition)

def Mesh(points, elements, geomEntity, boundary, dim):
    ''' Helper function to compute a new simplicial mesh ''' 
    return meshFromInfo(SimplicialMeshInfo(points, elements, geomEntity, boundary, dim))
