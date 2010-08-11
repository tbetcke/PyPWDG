import numpy

def gauss(n,intval=None):
    """Return tuple (x,w) of Gauss points and weights of order n in interval intval=(a,b).
       Default is intval=(0,1)
    """
    if (intval is None):
        intval=(0,1.)

    beta=.5/numpy.sqrt(1-(2*numpy.arange(1,n,dtype=numpy.float64))**(-2))
    T=numpy.diag(beta,1)+numpy.diag(beta,-1)
    (x,v)=numpy.linalg.eigh(T)
    I=numpy.argsort(x)
    x=x[I,:]
    x=intval[0]+(1.+x)/2*(intval[1]-intval[0])
    w=(intval[1]-intval[0])*v[0,I]**2
    return (x,w)

def tensorgauss(n,xintval=None,yintval=None):
    """Return tuple (xx,yy,ww) of (x,y) coordinates and corresponding weights
       for a tensor Gauss quadrature of order n in each coordinate direction.
       By default xintval=(0,1) and yintval=(0,1)
    """
    if (xintval is None):
        xintval=(0,1)
    if (yintval is None):
        yintval=(0,1)

    (x,wx)=gauss(n,xintval)
    (y,wy)=gauss(n,yintval)

    xx=numpy.zeros(n*n)
    yy=numpy.zeros(n*n)
    ww=numpy.zeros(n*n)

    for i,xelem in enumerate(x):
        xx[i*n:(i+1)*n]=xelem
        yy[i*n:(i+1)*n]=y
        ww[i*n:(i+1)*n]=wx[i]*wy

    return (xx,yy,ww)


