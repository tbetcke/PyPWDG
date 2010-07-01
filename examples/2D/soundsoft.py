import numpy
import scipy.special
from PyPWDG.PWDG2D import setup
from PyPWDG.PWDG2D.visualization import plotSol
from PyPWDG.PWDG2D import assembly
from PyPWDG.PWDG2D import bases
from PyPWDG.PWDG2D import solver



def ImpedanceBndFun(z,n,k):
    """ Plane Wave Impedance Boundary Condition
    
        z: Numpy array of m coordinates [x[j],y[j]], j=1,..,m
        n: Numpy array of 1 or m normal directions
                     [[n_x[0],..,n_x[m],
                      [n_y[0],..,n_y[m]]
        k: Wavenumber
    """
                      
    
    return numpy.zeros(len(z))


def planeWaveNeumannBndFun(z,n,k):
    """ Plane Wave Neumann Boundary Condition
    
        z: Numpy array of m coordinates [x[j],y[j]], j=1,..,m
        n: Numpy array of 1 or m normal directions
                     [[n_x[0],..,n_x[m],
                      [n_y[0],..,n_y[m]]
        k: Wavenumber
    """
                      
    
    d=numpy.array([1,1])
    d=d/numpy.linalg.norm(d)
    return -numpy.dot(d,n)*1j*k*numpy.exp(1j*k*numpy.dot(z,d))


def planeWaveDirichletBndFun(z,n,k):
    """ Plane Wave Dirichlet Boundary Condition
    
        z: Numpy array of m coordinates [x[j],y[j]], j=1,..,m
        n: Numpy array of 1 or m normal directions
                     [[n_x[0],..,n_x[m],
                      [n_y[0],..,n_y[m]]
        k: Wavenumber
    """
                      
    
    d=numpy.array([1,0])
    d=d/numpy.linalg.norm(d)
    return -numpy.exp(1j*k*numpy.dot(z,d))


def soundsourcebndfun(z,n,k):
    """ Sound source Boundary Condition
    
        z: Numpy array of m coordinates [x[j],y[j]], j=1,..,m
        n: Numpy array of 1 or m normal directions
                     [[n_x[0],..,n_x[m],
                      [n_y[0],..,n_y[m]]
        k: Wavenumber
    """
    
    source=numpy.array([-2.3, 0])
    zs=numpy.array([z[:,0]-source[0],z[:,1]-source[1]])
    dist=numpy.sqrt(zs[0,:]*zs[0,:]+zs[1,:]*zs[1,:])
    f=scipy.special.hankel1(0,k*dist)
    return -f


sqmesh='squarescatt.msh'
bndCond={10:('impedance',ImpedanceBndFun),
         11:('dirichlet',planeWaveDirichletBndFun)}
k=10
dgstruct=setup.init(sqmesh,bnd_cond=bndCond,
                    alpha=0.5,beta=0.5,delta=0.5,k=k,ngauss=20)

setup.addBasis(dgstruct,bases.PlaneWaveBasis,p=5,k=k)
print "Create local Vandermonde matrices"
assembly.processVandermonde(dgstruct)
print "Assemble fluxes"
A=assembly.assembleIntFlux(dgstruct)
(Abnd,rhs)=assembly.assembleBndFlux(dgstruct)
A=A+Abnd
dgstruct['A']=A
dgstruct['rhs']=rhs
solver.solveDirect(dgstruct)
print "Plot result"
plotSol(dgstruct,xrange=(-2,2),yrange=(-2,2),h=.05,plotMesh=False)
print "Complete"

