import numpy
import scipy.special
from PyPWDG.PWDG2D import setup
from PyPWDG.PWDG2D.visualization import plotSol
from PyPWDG.PWDG2D import assembly
from PyPWDG.PWDG2D import bases
from PyPWDG.PWDG2D import solver

def planeWaveImpedanceBndFun(z,n,k):
    """ Plane Wave Impedance Boundary Condition
    
        z: Numpy array of m coordinates [x[j],y[j]], j=1,..,m
        n: Numpy array of 1 or m normal directions
                     [[n_x[0],..,n_x[m],
                      [n_y[0],..,n_y[m]]
        k: Wavenumber
    """
                      
    
    d=numpy.array([1,1])
    d=d/numpy.linalg.norm(d)
    return (numpy.dot(d,n)-1)*1j*k*numpy.exp(1j*k*numpy.dot(z,d))

def planeWaveNeumannBndFun(z,n,k):
    """ Plane Wave Impedance Boundary Condition
    
        z: Numpy array of m coordinates [x[j],y[j]], j=1,..,m
        n: Numpy array of 1 or m normal directions
                     [[n_x[0],..,n_x[m],
                      [n_y[0],..,n_y[m]]
        k: Wavenumber
    """
                      
    
    d=numpy.array([1,1])
    d=d/numpy.linalg.norm(d)
    return numpy.dot(d,n)*1j*k*numpy.exp(1j*k*numpy.dot(z,d))


def planeWaveDirichletBndFun(z,n,k):
    """ Plane Wave Impedance Boundary Condition
    
        z: Numpy array of m coordinates [x[j],y[j]], j=1,..,m
        n: Numpy array of 1 or m normal directions
                     [[n_x[0],..,n_x[m],
                      [n_y[0],..,n_y[m]]
        k: Wavenumber
    """
                      
    
    d=numpy.array([1,1])
    d=d/numpy.linalg.norm(d)
    return numpy.exp(1j*k*numpy.dot(z,d))


def soundsourcebndfun(z,n,k):
    """ Sound source Boundary Condition
    
        z: Numpy array of m coordinates [x[j],y[j]], j=1,..,m
        n: Numpy array of 1 or m normal directions
                     [[n_x[0],..,n_x[m],
                      [n_y[0],..,n_y[m]]
        k: Wavenumber
    """
    
    source=numpy.array([-.2, 0])
    zs=numpy.array([z[:,0]-source[0],z[:,1]-source[1]])
    dist=numpy.sqrt(zs[0,:]*zs[0,:]+zs[1,:]*zs[1,:])
    f=scipy.special.hankel1(0,k*dist)
    df=-k*scipy.special.hankel1(1,k*dist)*zs/dist
    dfn=numpy.dot(n,df)
    return dfn-1j*k*f

#Load mesh and define boundary conditions
#Tuple numbers in boundary conditions refer to physical
#entitites in Gmsh.
sqmesh='square.msh'
bndCond={7:('impedance',planeWaveImpedanceBndFun),
         8:('dirichlet',planeWaveDirichletBndFun)}
#Define wavenumber
k=20
#Setup problem
dgstruct=setup.init(sqmesh,bnd_cond=bndCond,
                 alpha=0.5,beta=0.5,delta=0.5,k=k,ngauss=20)
setup.addBasis(dgstruct,bases.PlaneWaveBasis,p=10,k=k)
#Assemble matrices
assembly.processVandermonde(dgstruct)
A=assembly.assembleIntFlux(dgstruct)
(Abnd,b)=assembly.assembleBndFlux(dgstruct)
dgstruct['A']=A+Abnd
dgstruct['rhs']=b
# Solve and plot
solver.solveDirect(dgstruct)
plotSol(dgstruct)
print "Complete"


