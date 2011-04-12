import pypwdg.setup as ps
import pypwdg.core.bases as pcb
import pypwdg.mesh.mesh as pmm
import pypwdg.core.boundary_data as pcbd
import pypwdg.adaptivity.adaptivity as paa
import pypwdg.adaptivity.scripts as pas
import pypwdg.parallel.main

from numpy import array,sqrt

k = 30
direction=array([[1.0,1.0]])/sqrt(2)
g = pcb.PlaneWaves(direction, k)

bnddata={11:pcbd.zero_dirichlet(),
         10:pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g=g)}

bounds=array([[-2,2],[-2,2]],dtype='d')
npoints=array([200,200])

mesh = pmm.gmshMesh('squarescatt.msh',dim=2)

problem=ps.Problem(mesh,k,20, bnddata)
ibc = paa.InitialPWFBCreator(mesh,k,3,7)
pas.runadaptive(problem, ibc, "squarescatt", 6, bounds, npoints)

