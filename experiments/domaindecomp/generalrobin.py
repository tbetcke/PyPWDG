'''
Created on Jul 5, 2012

@author: joel
'''
import pypwdg.core.bases as pcb
import pypwdg.core.physics as pcp
import pypwdg.mesh.mesh as pmm
import pypwdg.mesh.structure as pms
import pypwdg.mesh.overlap as pmo
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
import scipy.sparse as ss
import math
import logging
log = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)
logging.getLogger('pypwdg.setup.domain').setLevel(logging.INFO)

@ppd.distribute()
class GeneralRobinPerturbation(object):
    
    def __init__(self, computationinfo, q):
        self.internalassembly = computationinfo.faceAssembly()
        mesh = computationinfo.problem.mesh
        cut = ss.spdiags((mesh.cutfaces > 0)*1, [0], mesh.nfaces,mesh.nfaces)
#        print cut
        self.B = q * (cut - cut * mesh.connectivity)
        self.Z = pms.AveragesAndJumps(mesh).Z     
        self.mesh = mesh   

    def getPerturbation(self):
        return pms.sumfaces(self.mesh, self.internalassembly.assemble([[self.B, self.Z], 
                                        [self.B, self.Z]]))
    

#@ppd.distribute()
#class GeneralRobinWorker(psd.SchwarzWorker):
#
#    def __init__(self, perturbation, mesh):
#        psd.SchwarzWorker.__init__(self, mesh)
#        self.perturbation = perturbation
#
#    @ppd.parallelmethod()
#    def initialise(self, system, sysargs, syskwargs):
#        extdofs = psd.SchwarzWorker.initialise(system, sysargs, syskwargs)
#        self.P = self.perturbation.getPerturbation()
#        overlapdofs = self.P.subrows()
#        log.info("Overlap dofs: %s"%overlapdofs) 
#        extdofs.append(overlapdofs)

@ppd.distribute()
class GeneralSchwarzWorker(object):
    def __init__(self, perturbation, mesh):
        self.mesh = mesh
        self.perturbation = perturbation
    
    @ppd.parallelmethod()
    def overlap(self):
        return [self.overlapdofs]
        
    @ppd.parallelmethod()
    def initialise(self, system, sysargs, syskwargs):
        """ Initialise the system in the worker
        
            returns a list of the neighbouring degrees of freedom required by this process
        """
        self.S,self.G = system.getSystem(*sysargs, **syskwargs) 
        self.P = self.perturbation.getPerturbation()
        self.overlapdofs = self.P.subrows()
        print self.overlapdofs.dtype
        log.info("Overlap dofs: %s"%self.overlapdofs) 
        return [self.S.subrows(self.mesh.neighbourelts), self.overlapdofs]
    
    @ppd.parallelmethod()
    def setexternalidxs(self, allextidxs):
        """ Tell this worker which degrees of freedom are exterior (i.e. are accessed by one process from another) 
        
            Returns this worker's contribution to the RHS for the Schur complement system
        """
        localidxs = self.S.subrows(self.mesh.partition) # the degrees of freedom calculated by this process
                                                        # N.B. for an overlapping method these arrays will overlap between processes -
                                                        # it still all works!
        sl = set(localidxs)
        extnooverlap = set(allextidxs).difference(self.overlapdofs)
        extidxs =  np.sort(np.array(list(sl.intersection(extnooverlap)), dtype=int)) # the exterior degrees for this process
        intidxs = np.array(list(sl.difference(extnooverlap)), dtype=int) # the interior degrees for this process
        self.intind = np.zeros(self.S.shape[0], dtype=bool) 
        self.intind[intidxs] = True # Create an indicator for the interior degrees

        log.debug("local %s"%localidxs)
        log.debug("external %s"%extidxs)
        log.debug("internal %s"%intidxs)

        M = self.S.tocsr() # Get CSR representations of the system matrix ...
        b = self.G.tocsr() # ... and the load vector
        P = self.P.tocsr()
        
        print "Non zero entries in perturbation matrix", sum(np.abs(P.data) > 1E-6)
        MpP = M + P
        MmP = M - P

        # Decompose the system matrix.  
        self.ext_allext = M[extidxs][:, allextidxs] 
        self.int_intinv = ssl.splu(MpP[intidxs][:,intidxs])
        self.int_allext = MmP[intidxs][:, allextidxs]
        self.ext_int = M[extidxs][:, intidxs]
        
        self.intsolveb = self.int_intinv.solve(b[intidxs].todense().A.squeeze())
        rhs = b[extidxs].todense().A.squeeze() - self.ext_int * self.intsolveb
        return [rhs]
        
        
    @ppd.parallelmethod()
    def multiplyext(self, x):
#        print x.shape, self.ext_allext.shape, self.ext_int.shape, self.int_allext.shape
        y = self.ext_allext * x - self.ext_int * self.int_intinv.solve(self.int_allext * x)
        return [y]  
    
    @ppd.parallelmethod()
    def precondext(self, x):        
        return x
#        return [self.DE.solve(x[self.localfromallext])]
    
    @ppd.parallelmethod(None, ppd.tuplesum)
    def recoverinterior(self, xe):
        """ Recover all the local interior degrees from the exterior degrees
        
            returns a tuple of the contribution to the global solution vector and an indicator of what 
            degrees have been written to.  The indicator is used to average duplicate contributions for 
            overlapping methods.
        """ 
        x = np.zeros_like(self.intind, dtype=complex)
        x[self.intind] = self.intsolveb - self.int_intinv.solve(self.int_allext * xe)
        return x, self.intind*1
        
   
import pypwdg.parallel.main

if __name__=="__main__":

    k = 15
    n = 4
    g = pcb.FourierHankel([-1,-1], [0], k)
    bdytag = "BDY"
    bnddata={bdytag:pcbd.dirichlet(g)}
    
    bounds=np.array([[0,1],[0,1]],dtype='d')
    npoints=np.array([200,200])
    mesh = tum.regularsquaremesh(n, bdytag)    
    
#    direction=np.array([[1.0,1.0]])/math.sqrt(2)
#    g = pcb.PlaneWaves(direction, k)
#    
#    bnddata={11:pcbd.zero_dirichlet(),
#             10:pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g=g)}
#    
#    bounds=np.array([[-2,2],[-2,2]],dtype='d')
#    npoints=np.array([200,200])
#    with puf.pushd('../../examples/2D'):
#        mesh = pmm.gmshMesh('squarescatt.msh',dim=2)


    basisrule = pcb.planeWaveBases(2,k,9)
    nquad = 7
   
#    mesh = pmo.overlappingPartitions(pmo.overlappingPartitions(mesh))
    mesh = pmo.overlappingPartitions(mesh)
    
   
#    problem = psp.Problem(mesh, k, bnddata)
    problem = psp.Problem(mesh,k,bnddata)
    
    compinfo = psc.ComputationInfo(problem, basisrule, nquad)
    computation = psc.Computation(compinfo, pcp.HelmholtzSystem)
    perturbation = GeneralRobinPerturbation(compinfo, 1E-6)

    sol = computation.solution(psd.GeneralSchwarzOperator(GeneralSchwarzWorker(perturbation, mesh)), psi.GMRESSolver('ctor'))
#    bs = psi.BrutalSolver(np.complex)
#    gsw1 = GeneralSchwarzWorker(perturbation, mesh)
#    sol = computation.solution(psd.GeneralSchwarzOperator(gsw1), bs)
#    M1 = bs.M
#    b1 = bs.b
#    x1 = bs.x
#    p2 = GeneralRobinPerturbation(compinfo, 1.0)
#    gsw2 = GeneralSchwarzWorker(p2, mesh)
#    sol = computation.solution(psd.GeneralSchwarzOperator(gsw2), bs)
#    
#    overlap1 = np.concatenate(gsw1.overlap())
#    overlap2 = np.concatenate(gsw2.overlap())
#    print overlap1
#    print overlap2
#    M2 = bs.M
#    b2 = bs.b
#    x2 = bs.x
#    dM = M1 - M2
#    print "dM", dM
#    mp.figure()
#    mp.spy(dM,markersize = 1)
#    print b1 - b2
#    print x1 - x2
#    print dM[9:12, :]
#    print dM[:,9:12]
#    print (x1 - x2)[9:12]
#    print sol.x
    
#    sol = computation.solution(SchwarzOperator(pmm.overlappingPartitions(mesh)), psi.GMRESSolver('ctor'))
    pom.outputMeshPartition(bounds, npoints, mesh)
    pom.output2dsoln(bounds, sol, npoints, show=True)