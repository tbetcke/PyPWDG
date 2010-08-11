import numpy

def plotSol(dgstruct,xrange=(0,1),yrange=(0,1),h=.01,plotMesh=True):
    """Plot the solution using Mayavi2

      Input parameters:

      xrange - Tuple (a,b) of grid boundaries in x-direction. Default (0,1)
      yrange - Tuple (c,d) of grid boundaries in y-direction. Default (0,1)
      plotMesh - Boolean. If True (default) also plot the mesh.
    """

    from pypwdg.GeometryTools.inpolygon import pnpoly

    dx=xrange[1]-xrange[0]
    dy=yrange[1]-yrange[0]
    nx=numpy.ceil(dx/h)
    ny=numpy.ceil(dy/h)
    
    xx=numpy.linspace(xrange[0],xrange[1],nx)
    yy=numpy.linspace(xrange[0],xrange[1],ny)
    X,Y=numpy.mgrid[xrange[0]:xrange[1]:1j*nx,yrange[0]:yrange[1]:1j*ny]
    s=X.shape
    X=X.flatten()
    Y=Y.flatten()
    F=numpy.zeros(nx*ny,dtype='c16')

    solvec=dgstruct['sol']

    
    nelems=dgstruct['nelems']
    points=dgstruct['mesh'].getAllNodes()

    for i in range(nelems):
        elem=dgstruct['mesh'].getElement(i)
        xpoints=[points[elem['nodes'][j]][0] for j in range(3)]
        ypoints=[points[elem['nodes'][j]][1] for j in range(3)]
        inpoly=pnpoly(X,Y,xpoints,ypoints)
        ind=numpy.nonzero(inpoly)
        for t in dgstruct['basfuns'][i]:
            vals=t[0].eval(X[ind],Y[ind])
            F[ind]+=numpy.dot(vals,solvec[t[2]])
        
    X=numpy.reshape(X,s)
    Y=numpy.reshape(Y,s)
    F=numpy.reshape(numpy.real(F),s)

    from enthought.mayavi import mlab

    elements=numpy.array([dgstruct['mesh'].getElement(i)['nodes'] for i in range(dgstruct['nelems'])])
    
    if plotMesh:
        m1=(mlab.triangular_mesh(points[:,0],points[:,1],numpy.zeros(len(points)),
                                 elements,representation='wireframe',color=(0,0,0)))
    m2=mlab.surf(X,Y,numpy.real(F),warp_scale=0)
    mlab.view(azimuth=0,elevation=0)
    mlab.show()
    
    

   
    
