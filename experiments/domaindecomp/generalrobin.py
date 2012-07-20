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
import pypwdg.parallel.decorate as ppd
import scipy.sparse.linalg as ssl
import pypwdg.parallel.mpiload as ppm
import pypwdg.test.utils.mesh as tum
import time
import numpy as np
import math
import logging
log = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)
logging.getLogger('pypwdg.setup.domain').setLevel(logging.INFO)

@ppd.distribute()
class GeneralRobinSystem(object):
    
    def __init__(self, computationinfo, overlapfaceentity, system):
        self.internalassembly = computationinfo.faceAssembly()
        mesh = computationinfo.problem.mesh
        self.B = mesh.connectivity * mesh.entityfaces[overlapfaceentity]
        self.system = system
        
    def getSystem(self):
        S,G = self.system.getSystem()
        self.internalassembly.assemble([[jk * self.alpha * AJ.JD,   -AJ.AN - B], 
                                            [AJ.AD + B,                -(self.beta / jk) * AJ.JN]])
    
   
import pypwdg.parallel.main

if __name__=="__main__":

    k = 15
#    n = 4
#    g = pcb.FourierHankel([-1,-1], [0], k)
#    bdytag = "BDY"
#    bnddata={bdytag:pcbd.dirichlet(g)}
#    
#    bounds=np.array([[0,1],[0,1]],dtype='d')
#    npoints=np.array([200,200])
#    mesh = tum.regularsquaremesh(n, bdytag)    
#    
    direction=np.array([[1.0,1.0]])/math.sqrt(2)
    g = pcb.PlaneWaves(direction, k)
    
    bnddata={11:pcbd.zero_dirichlet(),
             10:pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g=g)}
    
    bounds=np.array([[-2,2],[-2,2]],dtype='d')
    npoints=np.array([200,200])
    with puf.pushd('../../examples/2D'):
        mesh = pmm.gmshMesh('squarescatt.msh',dim=2)

    basisrule = pcb.planeWaveBases(2,k,9)
    nquad = 7
   
    mesh = pmm.overlappingPartitions(pmm.overlappingPartitions(mesh))
#    mesh = pmm.overlappingPartitions(mesh)
    
   
#    problem = psp.Problem(mesh, k, bnddata)
    problem = psp.Problem(pmm.overlappingPartitions(mesh),k,bnddata)
    
    compinfo = psc.ComputationInfo(problem, basisrule, nquad)
    computation = psc.Computation(compinfo, pcp.HelmholtzSystem)
    sol = computation.solution(psd.SchwarzOperator(mesh), psi.GMRESSolver('ctor'))
#    print sol.x
    
#    sol = computation.solution(SchwarzOperator(pmm.overlappingPartitions(mesh)), psi.GMRESSolver('ctor'))
    pom.outputMeshPartition(bounds, npoints, mesh)
    pom.output2dsoln(bounds, sol, npoints, show=True)