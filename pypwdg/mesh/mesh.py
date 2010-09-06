'''
Created on Jul 27, 2010

@author: tbetcke
'''
import numpy
from pypwdg.mesh.gmsh_reader import gmsh_reader
from pypwdg.utils.timing import *
import pymeshpart.mesh

class Mesh(object):
    """Mesh - Object that stores all necessary mesh information
       
       Usage:
       mesh=Mesh(meshDict,dim), where meshDict is a Dictionary returned
       from GmshReader and dim is either 2 for a 2D Mesh or 3 for a 3D Mesh
       
       Properties:

            gmsh_mesh     - Contains the mesh dictionary from the gmsh_reader
            faces         - List of Tuples (ind,vertices,v) defining the faces, where ind
                            is the associated element and vertices is a tuple containing the
                            defining vertices of the face. v contains the remaining non-face vertex
                            of the triangle/tetrahedron
            nfaces        - Number of faces = len(self.__faces). Faces are counted twice if they are between two elements
            nnodes        - Number of nodes
            nodes         - List of nodes
            facemap       - face1 is adjacent face2 if self.__facemap[face1]=face2. If self.__facemap[face1]==face1 the
                            face is on the boundary. face1,face2 are indices into the self.__faces list
            intfaces      - List of indices of faces in the interior
            bndfaces      - List of indices of faces on the boundary
            bnd_entities  - For each index i in self.__bndfaces self.__bnd_entities[i] is
                            the pysical entity of the boundary part assigned in Gmsh.
            nelements     - Number of elements
            elements      - List of elements
            dim           - Dimension of problem (dim=2,3)
            face_vertices - Number of vertices in face
            elem_vertices - Number of vertices in element
            normals       - Numpy array of dimension (self.nfaces,self.dim). self.__normals[ind] contains the normal
                            direction of the face with index ind.
            dets          - Numpy array of dimension self.nfaces, containing the absolute value of the cross product of the
                            partial derivatives for the map from the unit triangle (or unit line in 2d) to the face
            etof          - List of lists of faces for each element
            facepartitions- Partitioning of the faces (created by mesh.partitions)
            elempartitions- Partitioning of the elements (created by mesh.partitions)
            
            
    The elements and faces of a mesh are given a canonical ordering by self.__faces and self.__elements   
    """
    
    def __init__(self,mesh_dict,dim):
        """ Initialize Mesh
        
            The following private variables are created
            
            self.__gmsh_mesh     - Contains the mesh dictionary from the gmsh_reader
            self.__faces        - List of Tuples (ind,vertices,v) defining the faces, where ind
                                  is the index of the associated element in __elements and vertices is a tuple containing the
                                  defining vertices of the face. v contains the remaining non-face vertex
                                  of the triangle/tetrahedron
            self.__elements     - A list of all the elements (i.e. finite element domains) in __gmsh_mesh.   
            self.__nfaces       - Number of faces = len(self.__faces). Faces are counted twice if they are between two elements
            self.__facemap      - face1 is adjacent face2 if self.__facemap[face1]=face2. If self.__facemap[face1]==face1 the
                                  face is on the boundary. face1,face2 are indices into the self.__faces list
            self.__intfaces     - List of indices of faces in the interior
            self.__bndfaces     - List of indices of faces on the boundary
            self.__bnd_entities - For each index i in self.__bndfaces self.__bnd_entities[i] is
                                  the pysical entity of the boundary part assigned in Gmsh.
            self.__nelements        - Number of elements
            self.__dim             - Dimension of problem (dim=2,3)
            self.__face_vertices    - Number of vertices in face
            self.__elem_vertices    - Number of vertices in element
            self.__nnodes           - Number of nodes
            self.__facepartitions   - Partitioning of the faces (created by mesh.partitions)
            self.__elempartitions   - Partitioning of the elements (created by mesh.partitions)

        """
        from scipy.sparse import csr_matrix

        t = Timer().start()
        
        self.__gmsh_mesh=mesh_dict
        # Extract all faces and create face to vertex map
        t.split("Loaded file")
        
        gmshelems = self.__gmsh_mesh['elements']
        
        if dim==2:
            fv=2 # Number of vertices in faces
            ev=3 # Number of vertices in element
            gmsh_elem_key=2 # Key for triangle element in Gmsh 
            gmsh_face_key=1 # Key for line element in Gmsh
        elif dim==3:
            fv=3
            ev=4
            gmsh_elem_key=4 # Key for tetrahedal element in Gmsh
            gmsh_face_key=2 # Key for triangle element in Gmsh
            
        self.__dim=dim
        self.__face_vertices=fv
        self.__elem_vertices=ev

        # Pick out the coordinates of the vertices that we actually need
        self.__nodes = self.gmsh_mesh['nodes'][:,0:self.dim]
        self.__nnodes=len(self.__nodes)


        # we want a canonical ordering for the elements.
        self.__elements=filter(lambda e : e['type']==gmsh_elem_key, gmshelems.values())
        self.__nelements=len(self.__elements)
        
        # Faces are stored in the format (elem_key,(v1,..,vn)), where (v1,..,vn) define the face
        faces = ([(ekey,tuple(sorted(e['nodes'][0:i]+e['nodes'][i+1:ev])),e['nodes'][i]) for ekey, e in enumerate(self.__elements) for i in range(0,ev)])
        self.__faces=faces
        self.__nfaces=len(faces)       
        t.split("Created faces")
         
        # Create Face to Vertex Sparse Matrix
        ij=[[i for i in range(len(faces)) for j in range(fv)],
            [vs[1][i] for vs in faces for i in range(fv)]]         
        data=numpy.ones(len(ij[0]))
        ftov=csr_matrix((data,ij),dtype='i')
        ftof=ftov*ftov.transpose() # Multiply to get connectivity
        ftof=ftof.tocoo()
        (nzerox,nzeroy,vals)=(ftof.col,ftof.row,ftof.data)
        t.split("Done sparse stuff")
        indx=numpy.flatnonzero(vals==fv) # Get indices in connectivity matrices of adjacent faces        
#        facemap={} # facemap contains map from faces to adjacent faces (including reference to itself)
#        for ind in indx:
#                if not facemap.has_key(nzerox[ind]): facemap[nzerox[ind]]=[]
#                facemap[nzerox[ind]].append(nzeroy[ind])     
#        t.split("First bit of facemap")
##        print 'Start facemap'
#        for face1 in facemap:
#            if len(facemap[face1])==2: # Interior faces - delete reference to face itself
#                facemap[face1].remove(face1)
#            facemap[face1]=facemap[face1][0] # Extract the single element lists
#        t.split("Second bit of facemap")
#        self.__facemap=facemap
#
#
#        self.__intfaces=[ind for ind in range(len(faces)) if facemap[ind]!=ind]
#        self.__bndfaces=[ind for ind in range(len(faces)) if facemap[ind]==ind]
#        
#        t.split("int and bnd lists")

# This is about 10x faster than the above stuff.  Leaving it commented out for the moment just in case there's
# something that I've missed
        
        nonequal = nzerox[indx] != nzeroy[indx]
        faceids = numpy.arange(len(faces))
        facemap2 = faceids.copy()
        facemap2[nzerox[indx][nonequal]] = nzeroy[indx][nonequal]
        self.__facemap = facemap2
        self.__intfaces=faceids[facemap2 != faceids]
        self.__bndfaces=faceids[facemap2 == faceids]
        
        t.split("new facemap")
     
#         Create element to face map
        self.__etof=[[] for e in self.__elements] # note, this doesn't work:  [[]] * len(elements)
        for (iter,face) in enumerate(faces):
            self.__etof[face[0]].append(iter)

        t.split("etof")
                
        # Create map of physical entities
        self.__bnd_entities={}
        bnd_face_tuples = ([(gmshelems[key]['physEntity'],tuple(sorted(gmshelems[key]['nodes']))) for key in gmshelems if gmshelems[key]['type']==gmsh_face_key])
        tuple_entity_dict={}
        for elem in bnd_face_tuples:
            tuple_entity_dict[elem[1]]=elem[0]
        for ind in self.__bndfaces:
            self.__bnd_entities[ind]=tuple_entity_dict[self.__faces[ind][1]]
        
        t.split("entity maps")
        # Now generate direction vectors     
        self.__compute_directions()
        
        # Compute normals
        self.__compute_normals_and_dets()     
        t.split("directions and normals")
#        t.show()
        
#    @print_timing        
    def __compute_directions(self):
        """ Compute direction vectors for all faces 
        
            The following private variables are created
            
            self.__directions - three dimensional array, such that self.__directions[ind] returns
                                a matrix whose first row is the coordinate vector of the vertex v0. The two other
                                rows define the direction vectors v1-v0 and v2-v0. If the problem has two dimensional
                                the self.__directions[ind] only has two rows. 
        
        """
        # vertices is a 3-tensor.  For each (double sided) face, it contains a matrix of dim+1 coordinates.  The first dim of these
        # are on the face, the final one is the non-face vertex for the corresponding element
        vertices = numpy.array([[self.nodes[fv] for fv in face[1]] + [self.nodes[face[2]]] for face in self.faces])
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
                
            
    def reference_face_map(self,indx,xcoords,ycoords=None):
        """Return numpy array with transformed coordinates from reference element to face[indx]
        
            The unit triangle is defined as the triangle with coordinates (0,0), (1,0), (0,1)
        
        
           Input arguments:
           
           indx    - index of wanted face
           xcoords - numpy array with x coordinates in unit triangle
           ycoords - numpy array with y coordinate  in unit triangle (only needed for dim=3)
        
           Output arguments:
           
           coords - numpy array of dimension (dim,len(xcoords)), containing the result of the
                    map
                
        
        """
        coords=numpy.zeros((self.dim,len(xcoords)))
        if self.dim==2:
            coords[0,:]=self.__directions[indx,0,0]+self.__directions[indx,1,0]*xcoords
            coords[1,:]=self.__directions[indx,0,1]+self.__directions[indx,1,1]*xcoords
        else:
            coords[0,:]=self.__directions[indx,0,0]+self.__directions[indx,1,0]*xcoords+self.__directions[indx,2,0]*ycoords
            coords[1,:]=self.__directions[indx,0,1]+self.__directions[indx,1,1]*xcoords+self.__directions[indx,2,1]*ycoords
            coords[2,:]=self.__directions[indx,0,2]+self.__directions[indx,1,2]*xcoords+self.__directions[indx,2,2]*ycoords

        return coords  
            
    def partition(self,nparts):
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
             
    def get_gmsh_mesh(self):
        return self.__gmsh_mesh
    def get_faces(self):
        return self.__faces
    def get_nfaces(self):
        return self.__nfaces
    def get_facemap(self):
        return self.__facemap
    def get_intfaces(self):
        return self.__intfaces
    def get_bndfaces(self):
        return self.__bndfaces
    def get_bnd_entities(self):
        return self.__bnd_entities
    def get_nelements(self):
        return self.__nelements
    def get_elements(self):
        return self.__elements
    def get_dim(self):
        return self.__dim
    def get_face_vertices(self):
        return self.__face_vertices
    def get_elem_vertices(self):
        return self.__elem_vertices
    def get_normals(self):
        return self.__normals
    def get_dets(self):
        return self.__dets
    
    # Assign properties
    
    gmsh_mesh=property(get_gmsh_mesh)
    faces=property(get_faces)
    nfaces=property(get_nfaces)
    facemap=property(get_facemap)
    intfaces=property(get_intfaces)
    bndfaces=property(get_bndfaces)
    bnd_entities=property(get_bnd_entities)
    nelements=property(get_nelements)
    dim=property(get_dim)
    face_vertices=property(get_face_vertices)
    elem_vertices=property(get_elem_vertices)
    normals=property(get_normals)
    dets=property(get_dets)   
    directions = property(lambda self : self.__directions) 
    etof = property(lambda self: self.__etof)
    nodes = property(lambda self: self.__nodes)
    nnodes = property(lambda self: self.__nnodes)
    elements=property(get_elements)
    elempartitions=property(lambda self: self.__elempartitions)
    facepartitions=property(lambda self: self.__facepartitions)


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


    

        
    
        
    