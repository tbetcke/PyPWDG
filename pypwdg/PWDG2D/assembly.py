import numpy


def getnormal(p1,p2):
    """Compute a normal n for a line (p1,p2)"""
    n=numpy.array(p2)-numpy.array(p1)
    n=numpy.array([n[1],-n[0]])
    return n/numpy.linalg.norm(n)

def processVandermonde(dgstruct):
    """Create local Vandermonde matrices for all basis sets"""


    nodes=dgstruct['mesh'].getAllNodes()
    allFaces=dgstruct['mesh'].getAllFaces()
    vn=[[0,1],[1,2],[2,0]]
    x=dgstruct['gauss1d']['x']
    dgstruct['Vandermonde']={}

    for (elemNum,face) in allFaces:
        elem=dgstruct['mesh'].getElement(elemNum) 
        p0=nodes[elem['nodes'][vn[face][0]]]     
        p1=nodes[elem['nodes'][vn[face][1]]]
        xcoords=p0[0]+x*(p1[0]-p0[0])
        ycoords=p0[1]+x*(p1[1]-p0[1])
        nfunsElem=numpy.sum([t[1] for t in dgstruct['basfuns'][elemNum]])
        F=numpy.zeros((len(x),nfunsElem),dtype='c16')
        Fn=numpy.zeros((len(x),nfunsElem),dtype='c16')
        j=0
        n=getnormal(p0,p1)
        for (i,t) in enumerate(dgstruct['basfuns'][elemNum]):
            F[:,j:j+t[1]]=t[0].eval(xcoords,ycoords)
            Fn[:,j:j+t[1]]=t[0].evalNormal(xcoords,ycoords,n)
        dgstruct['Vandermonde'][(elemNum,face)]={'F':F,'Fn':Fn}


def assembleIntFlux(dgstruct):
    """ Return flux matrix A for interior edges"""

    nodes=dgstruct['mesh'].getAllNodes()
    vn=[[0,1],[1,2],[2,0]]
    k=dgstruct['k']
    nfuns=dgstruct['nfuns'] 
    from scipy import sparse
    A=sparse.csr_matrix((nfuns,nfuns),dtype='c16')
    
    w=dgstruct['gauss1d']['w']

    intFaces=dgstruct['mesh'].getIntFaces()
    EToE=dgstruct['mesh'].getEToE()
    EToF=dgstruct['mesh'].getEToF()


    alpha=dgstruct['flux_params']['alpha']
    beta=dgstruct['flux_params']['beta']
    for (elemNum,face) in intFaces:
        elem=dgstruct['mesh'].getElement(elemNum)
        (elem_p,face_p)=(EToE[elemNum,face],EToF[elemNum,face])
        p0=nodes[elem['nodes'][vn[face][0]]]     
        p1=nodes[elem['nodes'][vn[face][1]]]
        p=numpy.linalg.norm(p1-p0)
        # Get Vandermonde matrices
        F_m=dgstruct['Vandermonde'][(elemNum,face)]['F']
        Fn_m=dgstruct['Vandermonde'][(elemNum,face)]['Fn']
        F_p=dgstruct['Vandermonde'][(elem_p,face_p)]['F'][::-1]
        Fn_p=dgstruct['Vandermonde'][(elem_p,face_p)]['Fn'][::-1]
        U1=.5*F_m-beta/(1j*k)*Fn_m 
        U2=-.5*Fn_m+alpha*1j*k*F_m
        W=p*numpy.tile(w,(F_m.shape[1],1)).T
        C1=numpy.dot(Fn_m.conj().T,W*U1)
        C2=numpy.dot(Fn_p.conj().T,W*U1)
        C3=numpy.dot(F_m.conj().T,W*U2)
        C4=-numpy.dot(F_p.conj().T,W*U2)
        
        # Compute the correct indices
        g_m=[item for t in dgstruct['basfuns'][elemNum] for item in t[2]]
        g_p=[item for t in dgstruct['basfuns'][elem_p] for item in t[2]]
        g_mm=numpy.array([(i,j) for i in g_m for j in g_m]).transpose()
        g_mp=numpy.array([(i,j) for i in g_p for j in g_m]).transpose()
        
        # Assign the sparse matrices
        A=A+sparse.csr_matrix(((C1+C3).flatten(),g_mm),shape=(nfuns,nfuns),dtype='c16')
        A=A+sparse.csr_matrix(((C2+C4).flatten(),g_mp),shape=(nfuns,nfuns),dtype='c16')

    return A

def assembleBndFlux(dgstruct):
    """ Return rhs vector and flux matrix for boundary edges """

    nodes=dgstruct['mesh'].getAllNodes()
    vn=[[0,1],[1,2],[2,0]]
    k=dgstruct['k']
    x=dgstruct['gauss1d']['x']
    nfuns=dgstruct['nfuns'] 
    from scipy import sparse
    A=sparse.csr_matrix((nfuns,nfuns),dtype='c16')
    b=numpy.zeros(nfuns,'c16')    
    w=dgstruct['gauss1d']['w']

    bndFaces=dgstruct['mesh'].getBndFaces()


    delta=dgstruct['flux_params']['delta']

    for (elemNum,face) in bndFaces:
        elem=dgstruct['mesh'].getElement(elemNum)
        # Create matrices
        F=dgstruct['Vandermonde'][(elemNum,face)]['F']
        Fn=dgstruct['Vandermonde'][(elemNum,face)]['Fn']
        p0=nodes[elem['nodes'][vn[face][0]]]     
        p1=nodes[elem['nodes'][vn[face][1]]]
        p=numpy.linalg.norm(p1-p0)
        W=p*numpy.tile(w,(F.shape[1],1)).T
        # Find out correct boundary condition
        physId=dgstruct['mesh'].bndFaceEntity((elemNum,face))
        (bndCond,g)=dgstruct['bnd_cond'][physId]     
        # Evaluate g on boundary points
        xcoords=p0[0]+x*(p1[0]-p0[0])
        ycoords=p0[1]+x*(p1[1]-p0[1])
        z=numpy.array([xcoords,ycoords]).T
        gvals=g(z,getnormal(p0,p1),k)
        
        g_m=[item for t in dgstruct['basfuns'][elemNum] for item in t[2]]
        g_mm=numpy.array([(i,j) for i in g_m for j in g_m]).transpose()
 
        # Select flux formulation depending on boundary condition
        if bndCond=='impedance':
            U1=F-delta*(1/(1j*k)*Fn-F)
            U2=-Fn+(1-delta)*(Fn-1j*k*F)
            gvals1=gvals*delta/(1j*k)
            gvals2=(delta-1)*gvals
        elif bndCond=='dirichlet':
            U1=(1-delta)*F
            U2=-Fn+1j*k*(1-delta)*F
            gvals1=delta*gvals
            gvals2=(-1j*k)*(1-delta)*gvals
        elif bndCond=='neumann':
            U1=F-delta/(1j*k)*Fn
            U2=-Fn+(1-delta)*Fn
            gvals1=delta/(1j*k)*gvals
            gvals2=(delta-1)*gvals
        C=numpy.dot(Fn.conj().T,W*U1)+numpy.dot(F.conj().T,W*U2)
        A=A+sparse.csr_matrix((C.flatten(),g_mm),shape=(nfuns,nfuns),dtype='c16')
        b[g_m]+=-numpy.dot(w*gvals1,Fn.conj())-numpy.dot(w*gvals2,F.conj())

    return (A,b)


        
        
        
        
        
    
