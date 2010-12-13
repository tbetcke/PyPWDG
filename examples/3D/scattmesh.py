from pypwdg import setup,gmshMesh, PlaneWaves, planeWaveBases
from pypwdg import zero_impedance, dirichlet 
from numpy import array
import math

k = 10

g = PlaneWaves(array([[1,0,0]])/math.sqrt(1), k)

bnddata={82:zero_impedance(k), 83:dirichlet(g) }

bounds=array([[-2,2],[-2,2],[-2,2]],dtype='d')
npoints=array([100,100,100])

mesh = gmshMesh('scattmesh.msh',dim=3)
bases = planeWaveBases(mesh, k, 3)
comp=setup(mesh, k,16,bases,bnddata=bnddata,usecache=False)
comp.solve()
comp.writeSolution(bounds,npoints,fname='scattmesh.vti')
comp.writeMesh(fname='scattmesh.vtu',scalars=comp.combinedError())
