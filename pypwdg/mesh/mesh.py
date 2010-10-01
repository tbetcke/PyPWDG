'''
Created on Jul 27, 2010

@author: tbetcke
'''
import numpy
import numpy as np
from pypwdg.mesh.gmsh_reader import gmsh_reader
from pypwdg.utils.timing import *
import pymeshpart.mesh

def gmshMesh(gmsh_dict, dim):
    ''' Construct a Mesh from a gmsh dictionary '''
    
    if dim==2:
        gmsh_elem_key=2 # Key for triangle element in Gmsh 
        gmsh_face_key=1 # Key for line element in Gmsh
    elif dim==3:
        gmsh_elem_key=4 # Key for tetrahedal element in Gmsh
        gmsh_face_key=2 # Key for triangle element in Gmsh
    
    # Pick out the coordinates of the vertices that we actually need
    nodes = gmsh_dict['nodes'][:,0:dim]
    
    # This is the element->vertex map.  This establishes a canonical element ordering
    elements = map(lambda e: sorted(e['nodes']), filter(lambda e : e['type']==gmsh_elem_key, gmsh_dict['elements'].values()))
    
    # These are the physical entities in the mesh.
    boundaries = map(lambda f: (f['physEntity'], tuple(sorted(f['nodes']))), filter(lambda e : e['type']==gmsh_face_key, gmsh_dict['elements'].values()))
    return Mesh(nodes, elements, boundaries, dim)
        

class Mesh(object):
    """Mesh - The structure of a simplicial mesh
       
       Usage:
       mesh=Mesh(nodes, elements, boundaries, dim)
           nodes: a sequence of (dim)-coordinates, giving the vertices of the mesh
           elements: a sequence of sorted (dim+1)-tuples, giving the vertices for each element in the mesh
           boundaries: a sequence of tuples, each of the form (id, nodes), identifying the physical objects in the mesh produced by the mesh generator
           dim: the dimension of the mesh (2 or 3, although other numbers probably work too)
           
       
       Properties:

            faces         - For each face-element pair, a tuple of vertices.  The ordering reflects the ordering of elements.  Each internal face appears twice
            nonfacevertex - For each face-element pair, the element vertex which does not appear on that face
            nfaces        - Number of faces = len(self.faces). Faces are counted twice if they are between two elements
            nnodes        - Number of nodes
            nodes         - List of nodes
            facemap       - face1 is adjacent face2 if self.facemap[face1]=face2. If self.facemap[face1]==face1 the
                            face is on the boundary. face1,face2 are indices into the self.faces list
            intfaces      - List of indices of faces in the interior
            bndfaces      - List of indices of faces on the boundary
            boundaries    - a sequence of tuples, each of the form (id, nodes), identifying the physical objects in the mesh
            elements      - List of elements
            nelements     - Number of elements
            dim           - Dimension of problem (dim=2,3)

            normals       - Numpy array of dimension (self.nfaces,self.dim). self.__normals[ind] contains the normal
                            direction of the face with index ind.
            dets          - Numpy array of dimension self.nfaces, containing the absolute value of the cross product of the
                            partial derivatives for the map from the unit triangle (or unit line in 2d) to the face
            etof          - List of lists of faces for each element
            facepartitions- Partitioning of the faces (created by mesh.partitions)
            elempartitions- Partitioning of the elements (created by mesh.partitions)
            
            
    The elements and faces of a mesh are given a canonical ordering by self.__faces and self.__elements   
    """
    
    def __init__(self,nodes, elements, boundaries,dim):
        """ Initialize Mesh """
        from scipy.sparse import csr_matrix

        self.elements = elements
        self.nodes = nodes
        self.boundaries = boundaries
        self.dim = dim
        
        self.nnodes=len(nodes)
        self.nelements=len(elements)
        
        nev = dim+1
        
        # The vertices associated with each face
        faces = [e[0:i]+e[i+1:nev] for e in elements for i in range(0,nev)]
        self.faces = faces
        # The "opposite" vertex for each face    
        self.nonfacevertex = [e[i] for e in elements for i in range(0,nev)]    
        
        self.nfaces=len(faces)
        self.etof = np.arange(self.nfaces).reshape((-1,nev))
        
        ftov = csr_matrix((numpy.ones(self.nfaces * dim), numpy.concatenate(faces), np.arange(0, self.nfaces+1)*dim), dtype=int)
        ftov2=(ftov*ftov.transpose()).tocsr() # Multiply to get connectivity.
        ftof = csr_matrix((ftov2.data / dim, ftov2.indices, ftov2.indptr)) # It's an integer matrix, so dividing by dim means we're left with matching faces
        ftof.setdiag(np.zeros(self.nfaces))
        self.intfaces,otherfaces = ftof.nonzero()
        
        faceids = numpy.arange(len(faces))
        self.facemap = faceids.copy()
        self.facemap[self.intfaces] = otherfaces
        self.bndfaces=faceids[self.facemap == faceids]
        
        # Create map of physical entities
        self.bnd_entities={}
        for f in self.bndfaces:
            for id, bnodes in boundaries:
                if faces[f] in bnodes: self.bnd_entities[f]=id
        
        
        # Now generate direction vectors     
        self.__compute_directions()
        
        # Compute normals
        self.__compute_normals_and_dets()     
        
#    @print_timing        
    def __compute_directions(self):
        """ Compute direction vectors for all faces 
        
            The following private variables are created
            
            self.__directions - three dimensional array, such that self.__directions[ind] returns
                                a matrix:
                                    row 0 is the coordinate vector of the vertex v0. 
                                    rows 1:dim are the offsets to the other vertices of the face
                                    rows dim is the offset to the non-face vertex in the element 
        
        """
        # vertices is a 3-tensor.  For each (double sided) face, it contains a matrix of dim+1 coordinates.  The first dim of these
        # are on the face, the final one is the non-face vertex for the corresponding element
        vertices = numpy.array([[self.nodes[fv] for fv in fvs] + [self.nodes[otherv]] for fvs, otherv in zip(self.faces, self.nonfacevertex)])
        # M picks out the first coord and the differences to the others
        M = numpy.bmat([[numpy.mat([[1]]), numpy.zeros((1,self.dim))], [numpy.ones((self.dim,1))*-1, numpy.eye(self.dim)]])
        # Apply the differencing matrix to each set of coordinates
        dirs = numpy.tensordot(vertices, M, ([1],[1]))
        # Ensure that the directions live in the last dimension
        self.__directions = numpy.transpose(dirs, (0,2,1))
        
        # Set Partitions to None
        self.__facepartitions=None
        self.__elempartitions=None
            
#    @print_timing
    def __compute_normals_and_dets(self):
        """ Compute normal directions and determinants for all faces 
        
            The following private variables are created
            
            self.__normals - Numpy array of dimension (self.nfaces,self.dim). self.__normals[ind] contains the normal
                             direction of the face with index ind.
            self.__dets    - Numpy array of dimension self.nfaces, containing the absolute value of the cross product of the
                             partial derivatives for the map from the unit triangle (or line in 2d) to the face
        
        """
        
        self.__normals=numpy.zeros((self.nfaces,self.dim))
        
        if self.dim==2:
            # Put normal vectors 
            self.__normals[:,0]=self.__directions[:,1,1]
            self.__normals[:,1]=-self.__directions[:,1,0]
        else:
            self.__normals[:,0]=self.__directions[:,1,1]*self.__directions[:,2,2]-self.__directions[:,2,1]*self.__directions[:,1,2]
            self.__normals[:,1]=self.__directions[:,1,2]*self.__directions[:,2,0]-self.__directions[:,1,0]*self.__directions[:,2,2]
            self.__normals[:,2]=self.__directions[:,1,0]*self.__directions[:,2,1]-self.__directions[:,2,0]*self.__directions[:,1,1]

        # this is 100x faster than applying numpy.linalg.norm to each entry
        self.__dets = numpy.sqrt(numpy.sum(self.__normals * self.__normals, axis = 1))
                
        self.__normals *= (-numpy.sign(numpy.sum(self.__normals * self.__directions[:,-1,:], axis = 1)) / self.__dets ).reshape((-1,1))
                
 
            
    def partition(self,nparts):
        if nparts ==1:
             self.__facepartitions = [range(self.__nfaces)]
             self.__elempartitions = [range(self.__nelements)]
        else:
            """ Partition the mesh into nparts partitions """
            elemlist=[elem['nodes'] for elem in self.__elements]
            if self.__dim==2:
                elemtype=1
            else:
                elemtype=2
                             
            (epart,npart,edgecut)=pymeshpart.mesh.part_mesh_dual(elemlist,self.__nnodes,elemtype,nparts)
            facepartitions=[list() for p in range(nparts)]
            for i,face in enumerate(self.__faces): facepartitions[epart[face[0]]].append(i)
            self.__facepartitions=facepartitions
            self.__elempartitions=epart
    
        
        
            
    # Define get methods for properties
             
    def get_normals(self):
        return self.__normals
    def get_dets(self):
        return self.__dets
    
    # Assign properties
    
    normals=property(get_normals)
    dets=property(get_dets)   
    directions = property(lambda self : self.__directions) 
    elempartitions=property(lambda self: self.__elempartitions)
    facepartitions=property(lambda self: self.__facepartitions)


class MeshPart(object):
    pass
    

if __name__ == "__main__":

    print 'Import 2D mesh'
    import time
    t1=time.time()
    mesh_dict=gmsh_reader('../../examples/2D/square.msh')
    squaremesh=Mesh(mesh_dict,dim=2)
    t2=time.time()
    print 'Import took %0.3f seconds for mesh with %i elements' % (t2-t1,squaremesh.nelements)


    
    print 'Import 3D mesh'
    import time
    t1=time.time()
    mesh_dict=gmsh_reader('../../examples/3D/cube.msh')
    cubemesh=Mesh(mesh_dict,dim=3)
    t2=time.time()
    print 'Import took %0.3f seconds for mesh with %i elements' % (t2-t1,cubemesh.nelements)


    

        
    
        
    