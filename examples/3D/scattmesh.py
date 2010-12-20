import pypwdg.setup as ps
import pypwdg.core.bases as pcb
import pypwdg.mesh.mesh as pmm
import pypwdg.core.boundary_data as pcbd

from numpy import array
import math

k = 10

g = pcb.PlaneWaves(array([[1,0,0]])/math.sqrt(1), k)

bnddata={82:pcbd.zero_impedance(k), 83:pcbd.dirichlet(g) }

bounds=array([[-2,2],[-2,2],[-2,2]],dtype='d')
npoints=array([200,200,200])

mesh = pmm.gmshMesh('scattmesh.msh',dim=3)
bases = pcb.planeWaveBases(mesh, k, 3)
comp=ps.setup(mesh, k,16,bases,bnddata=bnddata,usecache=False)
comp.solve()
comp.writeSolution(bounds,npoints,fname='scattmesh.vti')
comp.writeMesh(fname='scattmesh.vtu',scalars=comp.combinedError())
