import numpy
from pypwdg.MeshReader.GmshReader import GmshReader

def Connect2D(elements,points):
    """Return the edges-to-edges and faces-to-faces matrices and sets of interior and boundary faces

    This routine is adapted from Connect2D.m in the book
    Nodal Discontinuous Galerkin Methods by Hesthaven and Warburton

    Keyword arguemnts:
    elements - A list of elements
    points - A list of points

    Output arguments:
    (EtoE, EtoF,int_faces,bnd_faces,all_faces) - Tuple of matrices EtoE and EtoF and sets int_faces, bnd_faces
                                       containing tuples (elem,face) of interior faces and boundary
                                       faces.
    """
    Nfaces=3

    K=len(elements)
    Nv=len(points)

    TotalFaces=Nfaces*K

    vn=[[0,1],[1,2],[2,0]]

    from scipy import sparse

    SpFToV=sparse.lil_matrix((TotalFaces,Nv),dtype='i')

    sk=0
    for elem in elements:
        for face in range(Nfaces):
            for ind in [elements[elem]['nodes'][i] for i in vn[face]]:
                SpFToV[sk,ind]=1
            sk+=1

    SpFToF=SpFToV*SpFToV.transpose()-2*sparse.eye(TotalFaces,TotalFaces,dtype='i')
    (nzerox,nzeroy)=SpFToF.nonzero()
    nzeros=zip(nzerox,nzeroy)
    faces=[(face1,face2) for (face1,face2) in nzeros if SpFToF[face1,face2]==2]

    element1=[f1 / 3 for (f1,f2) in faces]
    element2=[f2 / 3 for (f1,f2) in faces]

    face1=[f1 % 3 for (f1,f2) in faces]
    face2=[f2 % 3 for (f1,f2) in faces]


    EToE=numpy.outer(numpy.arange(K,dtype='i'),numpy.ones((Nfaces,1),dtype='i'))
    EToE[element1,face1]=element2

    EToF=numpy.outer(numpy.ones((K,1),dtype='i'),numpy.arange(Nfaces,dtype='i'))
    EToF[element1,face1]=face2

    all_faces=set(zip(range(K)*Nfaces,[0]*K+[1]*K+[2]*K))
    int_faces=set(zip(element1,face1))
    bnd_faces=all_faces.difference(int_faces)

    return (EToE,EToF,int_faces,bnd_faces,all_faces)


class TriangleMesh2D(object):
    """ Creates a 2D Triangle Mesh from a Gmsh file

    Constructor:
    TriangleMesh2D(filename) - filename is a valid Gmsh mesh file

    Methods:
    getElement(num)  - return dictionary with all information
    for the element with number num
    getAllFaces()    - return all faces
    getBndFaces()    - return list of tuples (elem,face) with all boundary
                       faces
    getIntFaces()    - return list of tuples (elem,face) with all interior
                       faces
    getNodes(sliceObj) - Return an array with the nodes specified by sliceObj,
    where sliceObj is a numpy slice
    bndFaceEntity(elemFace) -  Get the physical entitiy for a boundary face
                              tuple (elem,face)
    getIntEdges() - Return interior edges
    getBndEdges() - Return boundary faces
    getFaceNodes(elemFace) - Return nodes for a specific (elem,face) tuple
    getAllNodes() - Return all nodes
    getNelements() - Return number of triangle elements
    getEToE()      - Return Element to Element matrix
    getEToF()      - Return Element to Face matrix

    """
    def __init__(self,fname):
        meshDict=GmshReader(fname)
        self.nnodes_=meshDict['nnodes']
        # Process nodes (Throw away z-axis)        
        self.nodes_=numpy.array([numpy.array([p[0],p[1]]) for p in meshDict['nodes']])
        # Process elements
        self.elements_={}
        for elem in meshDict['elements']:
            if meshDict['elements'][elem]['type']==2:
                self.elements_[elem]=meshDict['elements'][elem]
        self.elemKeys_=self.elements_.keys()
        self.nelements_=len(self.elements_)

        (self.EToE_,self.EToF_,self.int_faces_,self.bnd_faces_,self.all_faces_)=Connect2D(self.elements_,self.nodes_)
        
        # Process edges

        self.int_edges_=set()
        self.bnd_edges_=set()
        for i in range(self.nelements_):
            for j in range(3):
                elem=self.EToE_[i,j]
                if i< elem:
                    self.int_edges_.add((i,elem,j,self.EToF_[i,j]))
                elif i==elem:
                    self.bnd_edges_.add((i,j))

        self.bndEdgeEntities_={}
        for elem in meshDict['elements']:
            if meshDict['elements'][elem]['type']==1:
                e0=meshDict['elements'][elem]['nodes'][0]                
                e1=meshDict['elements'][elem]['nodes'][1]
                if e0<=e1: 
                    t=(e0,e1)
                else:
                    t=(e1,e0)
                self.bndEdgeEntities_[t]=meshDict['elements'][elem]['physEntity']


        # Assign physical entities to boundary faces
        vn=[[0,1],[1,2],[2,0]]
        self.bndFaceEntities_={}
        for (elemNum,face) in self.bnd_faces_:
            elem=self.elements_[self.elemKeys_[elemNum]]
            p0=elem['nodes'][vn[face][0]]
            p1=elem['nodes'][vn[face][1]]
            if p0<=p1:
                t=(p0,p1)
            else:
                t=(p1,p0)
            self.bndFaceEntities_[(elemNum,face)]=self.bndEdgeEntities_.get(t,0)  


    def getElement(self,num):
        """ return dictionary with all information for the element with number num"""
        return self.elements_[self.elemKeys_[num]]
    def getAllFaces(self):
        """ return all faces"""
        return self.all_faces_
    def getBndFaces(self):
        """return list of tuples (elem,face) with all boundary faces"""
        return self.bnd_faces_
    def getIntFaces(self):
        """return list of tuples (elem,face) with all interior faces"""
        return self.int_faces_
    def getNodes(self,sliceObj):
        """Return an array with the nodes specified by sliceObj,
           where sliceObj is a numpy slice
        """
        return self.nodes_[sliceObj]
    def getAllNodes(self):
        """Return all nodes"""
        return self.nodes_
    def bndFaceEntity(self,elemFace):
        """Get the physical entitiy for a boundary face tuple (elem,face)"""
        return self.bndFaceEntities_[elemFace]
    
    def getIntEdges(self):
        """Return interior edges"""
        return self.int_edges_
    
    def getBndEdges(self):
        """Return boundary edges"""
        return self.bnd_edges_
    def getFaceNodes(self,elemFace):
        """Return nodes for a specific (elem,face) tuple"""
        vn=[[0,1],[1,2],[2,0]]
        (elem,face)=elemFace
        element=self.elements_[self.elemKeys_[elem]]
        p0=element['nodes'][vn[face][0]]
        p1=element['nodes'][vn[face][1]]
        return self.nodes_[[p0,p1]]
    def getNelements(self):
        """Return number of triangle elements"""
        return self.nelements_
    def getEToE(self):
        """Return EToE matrix"""
        return self.EToE_
    def getEToF(self):
        """Return EToF matrix"""
        return self.EToF_



