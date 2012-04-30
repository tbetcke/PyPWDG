'''
Created on Apr 30, 2012

@author: joel
'''
import numpy as np
import pypwdg.mesh.mesh as pmm
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
        return pmm.Mesh(self.nodes,self.elements,self.elemIdentity,self.boundaries,1)
    
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
    return pmm.Mesh(nodes, elements, elemIdentity, boundaries, 1)
    