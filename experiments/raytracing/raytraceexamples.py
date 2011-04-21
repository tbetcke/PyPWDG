'''
Created on Apr 20, 2011

@author: joel
'''
import pypwdg.core.bases as pcb
import pypwdg.mesh.mesh as pmm
import pypwdg.core.boundary_data as pcbd
import pypwdg.setup.problem as psp
import pypwdg.output.basis as pob
import pypwdg.raytrace.control as prc
import pypwdg.parallel.main

from numpy import array,sqrt

k = 30
#direction=array([[1.0,1.0]])/sqrt(2)
direction=array([[3,4]])/5.0
g = pcb.PlaneWaves(direction, k)

bnddata={11:pcbd.zero_dirichlet(),
         10:pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g=g)}

bounds=array([[-2,2],[-2,2]],dtype='d')
npoints=array([200,200])

mesh = pmm.gmshMesh('squarescatt.msh',dim=2)
problem = psp.Problem(mesh, k, bnddata)

etods = prc.tracemesh(problem, {10:lambda x:direction})
etob = [[pcb.PlaneWaves(array(ds), k)] for ds in etods]

pob.vtkbasis(mesh, etob, 'tracedirs.vtu', None)