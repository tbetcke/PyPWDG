import pypwdg.setup as ps
import pypwdg.core.bases as pcb
import pypwdg.mesh.mesh as pmm
import pypwdg.core.boundary_data as pcbd
import pypwdg.adaptivity.adaptivity as paa
import pypwdg.output.basis as pob
import pypwdg.utils.geometry as pug
import pypwdg.adaptivity.scripts as pas

import pypwdg.parallel.main
      
import numpy as np      
        
k = 60
direction=np.array([[1.0,1.0]])/np.sqrt(2)
#g = pcb.PlaneWaves(direction, k)
g = pcb.FourierHankel([-1,-1], [0], k)

impbd = pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g)

bnddata={7:impbd, 
         8:impbd}
#bnddata={7:pcbd.dirichlet(g), 
#         8:pcbd.dirichlet(g)}

bounds=np.array([[0,1],[0,1]],dtype='d')
npoints=np.array([250,250])

mesh = pmm.gmshMesh('square.msh',dim=2)
problem=ps.Problem(mesh,k,16, bnddata)
ibc = paa.InitialPWFBCreator(mesh,k,3,9)
pas.runadaptive(problem, ibc, "square", 20, bounds, npoints, g)