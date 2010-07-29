'''
Created on Jul 27, 2010

@author: tbetcke
'''
import numpy
from PyPWDG.mesh.gmsh_reader import gmsh_reader

class Mesh(object):
    """Mesh - Object that stores all necessary mesh information
       
       Usage:
       mesh=Mesh(meshDict,dim), where meshDict is a Dictionary returned
       from GmshReader and dim is either 2 for a 2D Mesh or 3 for a 3D Mesh
       
       Properties:

            gmsh_mesh     - Contains the mesh dictionary from the gmsh_reader
            faces         - List of Tuples (ind,vertices) defining the faces, where ind
                            is the associated element and vertices is a tuple containing the
                            defining vertices of the face
            nfaces        - Number of faces = len(self.__faces). Faces are counted twice if they are between two elements
            facemap       - face1 is adjacent face2 if self.__facemap[face1]=face2. If self.__facemap[face1]==face1 the
                            face is on the boundary. face1,face2 are indices into the self.__faces list
            intfaces      - List of indices of faces in the interior
            bndfaces      - List of indices of faces on the boundary
            bnd_entities  - For each index i in self.__bndfaces self.__bnd_entities[i] is
                            the pysical entity of the boundary part assigned in Gmsh.
            nelements     - Number of elements
       
       
    """
    
    def __init__(self,mesh_dict,dim):
        """ Initialize Mesh
        
            The following private variables are created
            
            self.__gmsh_mesh     - Contains the mesh dictionary from the gmsh_reader
            self.__faces        - List of Tuples (ind,vertices) defining the faces, where ind
                                  is the associated element and vertices is a tuple containing the
                                  defining vertices of the face
            self.__nfaces       - Number of faces = len(self.__faces). Faces are counted twice if they are between two elements
            self.__facemap      - face1 is adjacent face2 if self.__facemap[face1]=face2. If self.__facemap[face1]==face1 the
                                  face is on the boundary. face1,face2 are indices into the self.__faces list
            self.__intfaces     - List of indices of faces in the interior
            self.__bndfaces     - List of indices of faces on the boundary
            self.__bnd_entities - For each index i in self.__bndfaces self.__bnd_entities[i] is
                                  the pysical entity of the boundary part assigned in Gmsh.
            self.__nelements  - Number of elements
                             
            
            
        """
        
        self.__gmsh_mesh=mesh_dict
        # Extract all faces and create face to vertex map
        elems=self.__gmsh_mesh['elements']
     
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
            
        
        # Faces are stored in the format (elem_key,(v1,..,vn)), where (v1,..,vn) define the face
        faces = ([(key,tuple(sorted(elems[key]['nodes'][0:i]+elems[key]['nodes'][i+1:ev]))) for key in elems for i in range(0,ev) if elems[key]['type']==gmsh_elem_key])
        self.__faces=faces
        self.__nfaces=len(faces)       
         
        # Create Face to Vertex Sparse Matrix
        from scipy.sparse import csr_matrix
        ij=[[i for i in range(len(faces)) for j in range(fv)],
            [vs[1][i] for vs in faces for i in range(fv)]] 
        data=numpy.ones(len(ij[0]))
        ftov=csr_matrix((data,ij),dtype='i')
        ftof=ftov*ftov.transpose() # Multiply to get connectivity
        ftof=ftof.tocoo()
        (nzerox,nzeroy,vals)=(ftof.col,ftof.row,ftof.data)
        indx=numpy.flatnonzero(vals==fv) # Get indices in connectivity matrices of adjacent faces        
        facemap={} # facemap contains map from faces to adjacent faces (including reference to itself)
        for ind in indx:
                if not facemap.has_key(nzerox[ind]): facemap[nzerox[ind]]=[]
                facemap[nzerox[ind]].append(nzeroy[ind])     
        for face1 in facemap:
            if len(facemap[face1])==2: # Interior faces - delete reference to face itself
                facemap[face1].remove(face1)
            facemap[face1]=facemap[face1][0] # Extract the single element lists
        self.__facemap=facemap
        self.__intfaces=[ind for ind in range(len(faces)) if facemap[ind]!=ind]
        self.__bndfaces=[ind for ind in range(len(faces)) if facemap[ind]==ind]
                
        # Create element to face map
        self.__etof={}
        for (iter,face) in enumerate(faces):
            if face[0] in self.__etof:
                self.__etof[face[0]].append(iter)
            else:
                self.__etof[face[0]]=[iter]
        self.__nelements=len(self.__etof)
                
        # Create map of physical entities
        self.__bnd_entities={}
        bnd_face_tuples = ([(elems[key]['physEntity'],tuple(sorted(elems[key]['nodes']))) for key in elems if elems[key]['type']==gmsh_face_key])
        tuple_entity_dict={}
        for elem in bnd_face_tuples:
            tuple_entity_dict[elem[1]]=elem[0]
        for ind in self.__bndfaces:
            self.__bnd_entities[ind]=tuple_entity_dict[self.__faces[ind][1]]
     
        
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
    
    gmsh_mesh=property(get_gmsh_mesh)
    faces=property(get_faces)
    nfaces=property(get_nfaces)
    facemap=property(get_facemap)
    intfaces=property(get_intfaces)
    bndfaces=property(get_bndfaces)
    bnd_entities=property(get_bnd_entities)
    nelements=property(get_nelements)
    
        
        
        

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
 


        
    
        
    