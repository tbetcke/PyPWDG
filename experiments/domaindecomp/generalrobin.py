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
import scipy.sparse.linalg as ssl
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
    

@ppd.distribute()
class PerturbedSchwarzWorker(psd.SchwarzWorker):
    def __init__(self, perturbation, mesh):
        psd.SchwarzWorker.__init__(self, mesh)
        self.perturbation = perturbation
    
    @ppd.parallelmethod()
    def overlap(self):
        return [self.overlapdofs]
        
    @ppd.parallelmethod()
    def initialise(self, system, sysargs, syskwargs):
        neighbours = psd.SchwarzWorker.initialise(self, system, sysargs, syskwargs)
        self.P = self.perturbation.getPerturbation()
        self.overlapdofs = self.P.subrows()
        log.info("Overlap dofs: %s"%self.overlapdofs) 
        return neighbours+[self.overlapdofs]
    
    @ppd.parallelmethod()
    def setexternalidxs(self, allextidxs):
        """ Todo: refactor the base class to avoid duplication here
        """
        localidxs = self.S.subrows(self.mesh.partition) # the degrees of freedom calculated by this process
                                                        # N.B. for an overlapping method these arrays will overlap between processes -
                                                        # it still all works!
        sl = set(localidxs)
        extnooverlap = set(allextidxs).difference(self.overlapdofs)
        extidxs =  np.sort(np.array(list(sl.intersection(extnooverlap)), dtype=int)) # the exterior degrees for this process
        intidxs = np.sort(np.array(list(sl.difference(extnooverlap)), dtype=int)) # the interior degrees for this process
        self.intind = np.zeros(self.S.shape[0], dtype=bool) 
        self.intind[intidxs] = True # Create an indicator for the interior degrees
        self.localext = allextidxs.searchsorted(extidxs)

        log.info("local %s"%localidxs)
        log.info("external %s"%extidxs)
        log.info("internal %s"%intidxs)
        log.info("localext %s"%self.localext)

        M = self.S.tocsr() # Get CSR representations of the system matrix ...
        b = self.G.tocsr() # ... and the load vector
        P = self.P.tocsr()
        
        log.info("Non zero entries in perturbation matrix = %s"%sum(np.abs(P.data) > 1E-6))
        MpP = M + P
        MmP = M - P

        # Decompose the system matrix.  
        self.ext_allext = M[extidxs][:, allextidxs] 
        self.int_intinv = ssl.splu(MpP[intidxs][:,intidxs])
        self.int_allext = MmP[intidxs][:, allextidxs]
        self.ext_int = M[extidxs][:, intidxs]
#        self.ext_extinv = ssl.splu(MpP[extidxs, :][:, extidxs])
        
#        mp.subplot(1,3,1)
#        mp.spy(self.ext_allext, markersize=1)
#        mp.subplot(1,3,2)
#        mp.spy(self.int_allext, markersize=1)
#        mp.subplot(1,3,3)
#        mp.spy(self.ext_int, markersize=1)
#        mp.show()
        
        self.intsolveb = self.int_intinv.solve(b[intidxs].todense().A.squeeze())
        rhs = b[extidxs].todense().A.squeeze() - self.ext_int * self.intsolveb
        return [rhs]        
#
#@ppd.parallel()
#def test():
#    import time
#    time.sleep(np.random.rand() / 10)
#    return [ppd.comm.rank]
   
import pypwdg.parallel.main

def search():
    k = 5
    g = pcb.FourierHankel([-1,-1], [0], k)
    bdytag = "BDY"
    bnddata={bdytag:pcbd.dirichlet(g)}
    
    bounds=np.array([[0,1],[0,1]],dtype='d')
    npoints=np.array([200,200])
    relerr = []
    params = []
    for n in range(4,10):
        mesh = tum.regularsquaremesh(n, bdytag)    
        mesh = pmo.overlappingPartitions(mesh)
        problem = psp.Problem(mesh,k,bnddata)
    #    meshinfo = tum.regularsquaremeshinfo(n, bdytag)
    #    topology = pmm.Topology(meshinfo)
    #    partition = pmm.BespokePartition(meshinfo, topology, lambda n: np.arange(meshinfo.nelements).reshape(n, -1))    
    #    mesh = pmm.MeshView(meshinfo, topology, partition)
        
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
    
    
        # goes wrong for n=7 with npw = 6 & 7.
        for npw in range(1, 12):  
            basisrule = pcb.planeWaveBases(2,k,npw)
            nquad = 10
           
        #    mesh = pmo.overlappingPartitions(pmo.overlappingPartitions(mesh))
            
           
        #    problem = psp.Problem(mesh, k, bnddata)
            
            compinfo = psc.ComputationInfo(problem, basisrule, nquad)
            computation = psc.Computation(compinfo, pcp.HelmholtzSystem)
            sold = computation.solution(psc.DirectOperator(), psc.DirectSolver())
            pom.output2dsoln(bounds, sold, npoints, show = False)
            for p in [0]:#, 1E-6, 1]:# 1j, -0.1, -0.1j]:
                perturbation = GeneralRobinPerturbation(compinfo, p)
            
                op = psd.GeneralSchwarzOperator(PerturbedSchwarzWorker(perturbation, mesh))
        #        sol = computation.solution(op, psi.GMRESSolver('ctor'))
                sol = computation.solution(op, psi.BrutalSolver(np.complex))       
                ds = np.abs(sold.x - sol.x)
                relerr.append(np.max(ds) / np.max(np.abs(sold.x)))
                params.append((n,npw))
                print relerr
                print params

if __name__=="__main__":
#    search()
#    exit()
    k = 1
    n = 6
    g = pcb.FourierHankel([-1,-1], [0], k)
    bdytag = "BDY"
    bnddata={bdytag:pcbd.dirichlet(g)}
    
    bounds=np.array([[0,1],[0,1]],dtype='d')
    npoints=np.array([200,200])
#    mesh = tum.regularsquaremesh(n, bdytag)    
    meshinfo = tum.regularsquaremeshinfo(n, bdytag)
    topology = pmm.Topology(meshinfo)
    partition = pmm.BespokePartition(meshinfo, topology, lambda n: np.arange(meshinfo.nelements).reshape(n, -1))    
    mesh = pmm.MeshView(meshinfo, topology, partition)
    
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


    # goes wrong for n=7 with npw = 6 & 7.  
    npw = 2  
    basisrule = pcb.planeWaveBases(2,k,npw)
    nquad = 10
   
#    mesh = pmo.overlappingPartitions(pmo.overlappingPartitions(mesh))
    mesh = pmo.overlappingPartitions(mesh)
    
   
#    problem = psp.Problem(mesh, k, bnddata)
    problem = psp.Problem(mesh,k,bnddata)
    
    compinfo = psc.ComputationInfo(problem, basisrule, nquad)
    computation = psc.Computation(compinfo, pcp.HelmholtzSystem)
    sold = computation.solution(psc.DirectOperator(), psc.DirectSolver())
    pom.output2dsoln(bounds, sold, npoints, show = False)
    for p in [0]:#, 1E-6, 1]:# 1j, -0.1, -0.1j]:
        perturbation = GeneralRobinPerturbation(compinfo, p)
    
        op = psd.GeneralSchwarzOperator(PerturbedSchwarzWorker(perturbation, mesh))
#        sol = computation.solution(op, psi.GMRESSolver('ctor'))
        sol = computation.solution(op, psi.BrutalSolver(np.complex))       
        ds = np.abs(sold.x - sol.x)
        mds = np.min(ds)
        print mds
#        print np.log10(ds / mds).astype(int).reshape(-1,npw)
        pom.output2dsoln(bounds, sol, npoints, show=False)
        print np.hstack((sol.x.reshape(-1,npw), sold.x.reshape(-1,npw)))
        n = len(op.rhs())
        M = np.hstack([op.multiply(x).reshape(-1,1) for x in np.eye(n)])
        e = np.linalg.eigvals(M)
#        print e
        pom.mp.figure()
        pom.mp.scatter(e.real, e.imag)
                 
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
    
    pom.mp.show()
    pom.output2dsoln(bounds, sol, npoints, show=True)