'''
Created on Jun 1, 2012

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
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger('pypwdg.setup.indirect').setLevel(logging.INFO)

import pypwdg.parallel.main


def solveMortar(problem, basisrule, mortardegree, nquad, system, solver):
    mortarrule = pcbr.ReferenceBasisRule(pcbr.Legendre1D(mortardegree))
    s = -1j*k
    mc = psm.MortarComputation(problem, basisrule, mortarrule, nquad, system, system.boundaryclass, s)
    return mc.solution(solver, dovolumes=True)     
     
def compare(problem, basisrule, mortardegree, nquad, system, plotdata = None):
    if plotdata:
        bounds, npoints = plotdata
    it = psi.ItTracker()
    solver = psi.GMRESSolver('ctor', it)
    sm = solveMortar(problem, basisrule, 2, nquad, system, solver)
    if plotdata: pom.output2dsoln(bounds, sm, npoints, show=False)
    itsm = np.array(it.reset())
    
    compinfo = psc.ComputationInfo(problem, basisrule, nquad)
    computation = psc.Computation(compinfo, system)

    sdd = computation.solution(psd.DomainDecompOperator(problem.mesh), solver)
    if plotdata: pom.output2dsoln(bounds, sdd, npoints, show=False)
    itsdd = np.array(it.reset())
    sb = computation.solution(psi.BlockPrecondOperator(problem.mesh), solver)
    if plotdata: pom.output2dsoln(bounds, sb, npoints, show=False)
    itsb = np.array(it.reset())
    
    mp.figure()
    mp.hold(True)
    mp.semilogy(itsm, 'b')
#    mp.figure()
    mp.semilogy(itsdd, 'r')
#    mp.figure()
    mp.semilogy(itsb, 'g')
    mp.show()
#    print itsm, itsdd, itsb


if __name__=="__main__":
    k = 10
    direction=np.array([[1.0,1.0]])/math.sqrt(2)
    g = pcb.PlaneWaves(direction, k)
    
    bnddata={11:pcbd.zero_dirichlet(),
             10:pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g=g)}
    
    bounds=np.array([[-2,2],[-2,2]],dtype='d')
    npoints=np.array([200,200])
    with puf.pushd('../../examples/2D'):
        mesh = pmm.gmshMesh('squarescatt.msh',dim=2)
    basisrule = pcb.planeWaveBases(2,k,11)
    nquad = 5
    problem = psp.Problem(mesh, k, bnddata)
    
    compare(problem, basisrule, 2, nquad, pcp.HelmholtzSystem)#, (bounds, npoints))
    
