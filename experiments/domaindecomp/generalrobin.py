'''
Created on Jul 5, 2012

@author: joel
'''
import pypwdg.core.bases as pcb
import pypwdg.core.physics as pcp
import pypwdg.mesh.mesh as pmm
import pypwdg.core.boundary_data as pcbd
import pypwdg.utils.file as puf
import pypwdg.setup.problem as psp
import pypwdg.setup.computation as psc
import pypwdg.setup.mortar as psm
import pypwdg.setup.indirect as psi
import pypwdg.setup.domain as psd
import pypwdg.core.bases.reference as pcbr
import matplotlib.pyplot as mp
import pypwdg.output.mploutput as pom
import numpy as np
import math
import logging
logging.getLogger().setLevel(logging.INFO)
#logging.getLogger('pypwdg.setup.indirect').setLevel(logging.DEBUG)

import pypwdg.parallel.main

if __name__=="__main__":
    k = 20
    direction=np.array([[1.0,1.0]])/math.sqrt(2)
    g = pcb.PlaneWaves(direction, k)
    
    bnddata={11:pcbd.zero_dirichlet(),
             10:pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g=g)}
    
    bounds=np.array([[-2,2],[-2,2]],dtype='d')
    npoints=np.array([200,200])
    with puf.pushd('../../examples/2D'):
        mesh = pmm.gmshMesh('squarescatt.msh',dim=2)
    basisrule = pcb.planeWaveBases(2,k,15)
    nquad = 7
   
    problem = psp.Problem(mesh, k, bnddata)
    compinfo = psc.ComputationInfo(problem, basisrule, nquad)
    computation = psc.Computation(compinfo, pcp.HelmholtzSystem)
    sol = computation.solution(psc.DirectOperator(), psc.DirectSolver())
    pom.output2dsoln(bounds, sol, npoints, show=True)