'''
Created on Jul 27, 2010

@author: tbetcke
'''
import numpy
from PyPWDG.mesh.gmsh_reader import gmsh_reader

class Mesh(object):
    """Mesh - Object that stores all necessary mesh information
       
       Usage:
       mesh=Mesh(meshDict), where meshDict is a Dictionary returned
       from GmshReader
       
       
    """
    
    def __init__(self,mesh_dict):
        """ Initialize Mesh
        
            The following private variables are created
            
            self.__mesh - Contains the mesh dictionary from the gmsh_reader
            self.__faces - List of Tuples (ind,vertices) defining the faces, where ind
                           is the associated element and vertices is a tuple containing the
                           defining vertices of the face
            self.__nfaces - Number of faces = len(self.__faces). Faces are counted twice if they are between two elements
            self.__facemap - face1 is adjacent face2 if self.__facemap[face1]=face2. If self.__facemap[face1]==face1 the
                             face is on the boundary. face1,face2 are indices into the self.__faces list
            self.__intfaces - List of indices of faces in the interior
            self.__bndfaces - List of indices of faces on the boundary
                             
            
            
        """
        
        self.__mesh=mesh_dict
        # Extract all faces and create face to vertex map
        elems=self.__mesh['elements']
     
        
        # Faces are stored in the format (elem_key,(v1,..,vn)), where (v1,..,vn) define the face
        print 'Create faces'
        faces = ([(key,tuple(sorted(elems[key]['nodes'][0:i]+elems[key]['nodes'][i+1:4]))) for key in elems for i in range(0,4) if elems[key]['type']==4])
        self.__faces=faces
        self.__nfaces=len(faces)        
        # Create Face to Vertex Sparse Matrix
        print 'Create sparse matrix'
        from scipy.sparse import csr_matrix
        ij=[[i for i in range(len(faces)) for j in range(3)],
            [vs[1][i] for vs in faces for i in range(3)]] # vs[i]-1 since nodes should start at zero
        data=numpy.ones(len(ij[0]))
        ftov=csr_matrix((data,ij),dtype='i')
        ftof=ftov*ftov.transpose()
        ftof=ftof.tocoo()
        print 'Sparse matrix evaluated'
        (nzerox,nzeroy,vals)=(ftof.col,ftof.row,ftof.data)
        indx=numpy.flatnonzero(vals==3)        
        facemap={}
        for ind in indx:
                if not facemap.has_key(nzerox[ind]): facemap[nzerox[ind]]=[]
                facemap[nzerox[ind]].append(nzeroy[ind])
        print 'Second connections for loop'        
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
                
        # Create map of physical entities
        print "Create physical entities"
        self.__bnd_entities={}
        bnd_face_tuples = ([(elems[key]['physEntity'],tuple(sorted(elems[key]['nodes']))) for key in elems for i in range(0,4) if elems[key]['type']==2])
        tuple_entity_dict={}
        for elem in bnd_face_tuples:
            tuple_entity_dict[elem[1]]=elem[0]
        for ind in self.__bndfaces:
            self.__bnd_entities[ind]=tuple_entity_dict[self.__faces[ind][1]]
     
        
        
        
        
        

if __name__ == "__main__":
    print 'Start'
    mesh_dict=gmsh_reader('../../examples/3D/cube_fine.msh')
    cubemesh=Mesh(mesh_dict)
    print 'Success'

     

        
    
        
    