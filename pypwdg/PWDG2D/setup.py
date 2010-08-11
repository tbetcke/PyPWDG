import numpy
from pypwdg.PWDG2D.integration import gauss, tensorgauss
from pypwdg.MeshReader.Triangle2D import TriangleMesh2D

def addBasis(dgstruct,btype,**kwargs):
    """ Add a set of basis functions to dgstruct

        btype - A class object defining the basis
        kwargs - Keyword Arguments for the constructor of the basis objects
                 At least one of kwargs must be 'p' defining the number of
                 basis functions

    """

    from numpy import exp,array,real,imag

    nelems=dgstruct['nelems']

    if not dgstruct.has_key('basfuns'):
        dgstruct['basfuns']=dict.fromkeys(range(nelems))
        dgstruct['nfuns']=0

    nfuns=dgstruct['nfuns']
    p=kwargs['p']

    for elem in range(nelems):
        if dgstruct['basfuns'][elem] is None:
            dgstruct['basfuns'][elem] = []
        dgstruct['basfuns'][elem].append((btype(**kwargs),p,nfuns+numpy.arange(p)))
        nfuns+=p

    dgstruct['nfuns']=nfuns

def assignRefrIndex(dgstruct,refr_ind):
    """ Assign a refractive index function to a problem and compute for each element the
        average of the refractive index on the element

        Arguments:

        refr_ind - A pointer to a function f(x,y)
    """
    dgstruct['refr_ind']['fun']=refr_ind
    dgstruct['refr_ind']['piecewise_const']=numpy.ones(dgstruct['nelems'])

    nelems=dgstruct['nelems']
    xx=dgstruct['gauss2d']['x']
    yy=dgstruct['gauss2d']['y']
    ww=dgstruct['gauss2d']['w']

    for elem in range(nelems):
        # Get coordinates of the element
        nodeids=dgstruct['mesh'].getElement(elem)['nodes']
        nodes=dgstruct['mesh'].getNodes(nodeids)
        pd1=nodes[1]-nodes[0]
        pd2=nodes[2]-nodes[0]
        x=xx*(1-yy)
        y=yy
        xcoords=nodes[0,0]+x*(nodes[1,0]-nodes[0,0])+y*(nodes[2,0]-nodes[0,0])
        ycoords=nodes[0,1]+x*(nodes[1,1]-nodes[0,1])+y*(nodes[2,1]-nodes[0,1])
        feval=(1-yy)*refr_ind(xcoords,ycoords)
        dgstruct['refr_ind']['piecewise_const'][elem]=2.0*numpy.dot(feval,ww)


def init(mesh,bnd_cond,k=5,alpha=.5,beta=.5,delta=.5,refr_ind=lambda x,y: 1,ngauss=5):
    """Initialize a wave problem

    Input parameters:
    mesh      - filename of Gmsh Mesh 
    bnd_cond  - Structure defining the boundary conditions 
    k         - wavenumber
    alpha     - flux parameter
    beta      - flux parameter
    delta     - flux parameter
    refr_ind  - refractive index function
    ngauss    - number of integration points in each direction

    Output parameters:
    dgstruct - data structure containing all necessary problem information
    """


    dgstruct={}
    dgstruct['k']=k
    dgstruct['mesh']=TriangleMesh2D(mesh)
    dgstruct['nelems']=dgstruct['mesh'].getNelements()
    
    print "Loaded mesh with %d elements" % dgstruct['nelems']
    
    dgstruct['refr_ind']={}
    dgstruct['bnd_cond']=bnd_cond
    dgstruct['sol']=None
    
    dgstruct['flux_params']={'alpha':alpha,'beta':beta,'delta':delta}
 
    (x,w)=gauss(ngauss)
    dgstruct['gauss1d']={'x':x,'w':w}

    (xx,yy,ww)=tensorgauss(ngauss)
    dgstruct['gauss2d']={'x':xx,'y':yy,'w':ww}
    assignRefrIndex(dgstruct,refr_ind)
    return dgstruct

        
        

