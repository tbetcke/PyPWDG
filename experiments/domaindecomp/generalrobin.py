'''
Created on Jul 5, 2012

@author: joel
'''
import pypwdg.core.bases as pcb
import pypwdg.core.bases.reference as pcbr
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
import scipy.io as sio
import math
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger('pypwdg.setup.domain').setLevel(logging.INFO)

@ppd.distribute()
class GeneralRobinPerturbation(object):
    
    def __init__(self, computationinfo, q):
        self.internalassembly = computationinfo.faceAssembly()
        mesh = computationinfo.problem.mesh
        AJ = pms.AveragesAndJumps(mesh)
        cut = ss.spdiags((mesh.cutfaces > 0)*1, [0], mesh.nfaces,mesh.nfaces)
        print 'cut', cut
#        self.B = q * cut * AJ.JD
#        self.B = q * (cut - cut * mesh.connectivity)
        self.B = q * (cut * mesh.connectivity)
        self.Z = AJ.Z     
        self.mesh = mesh   

    def getPerturbation(self):
        return pms.sumfaces(self.mesh, self.internalassembly.assemble([[self.B, self.Z], 
                                                                       [self.B, self.Z]]))
    
    def getNeighbours(self):
        return self.mesh.part.oldneighbours
    

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
        self.overlapdofs = self.P.subrows(self.perturbation.getNeighbours())
        log.debug("Overlap dofs: %s"%self.overlapdofs) 
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
        
        intminusidxs = np.sort(np.array(list(sl.difference(allextidxs)), dtype=int))
        self.intind = np.zeros(self.S.shape[0], dtype=bool) 
        self.intind[intminusidxs] = True # Create an indicator for the interior degrees
        self.intminusint = intidxs.searchsorted(intminusidxs)
        
        self.localext = allextidxs.searchsorted(extidxs)
#        print "localext", self.localext
#        print "intminusint", self.intminusint

        log.debug("local %s"%localidxs)
        log.debug("external %s"%extidxs)
        log.debug("internal %s"%intidxs)
        log.debug("localext %s"%self.localext)
        
        log.debug("overlap %s"%self.overlapdofs)

        M = self.S.tocsr() # Get CSR representations of the system matrix ...
        b = self.G.tocsr() # ... and the load vector
        P = self.P.tocsr()
#        print "M", M.todense()
#        print "P", P.todense()
#        print "b", b.todense()
        
        log.info("Non zero entries in perturbation matrix = %s"%sum(np.abs(P.data) > 1E-6))
#        print "Pinternal", P[intidxs]
#        print "Pei", P[extidxs][:, intidxs]
#        print "Pee", P[extidxs][:, allextidxs]
        MpP = M + P
        MmP = M - P

        # Decompose the system matrix.  
        self.ext_allext = MmP[extidxs][:, allextidxs] 
        self.int_intinv = ssl.splu(M[intidxs][:,intidxs])
        self.int_allext = M[intidxs][:, allextidxs]
        self.ext_int = MpP[extidxs][:, intidxs]
        self.int_ext = M[intidxs][:, extidxs]
#        pom.mp.spy(M[intidxs][:,intidxs], markersize=1)
#        pom.mp.figure()
#        pom.mp.spy(self.int_intinv.solve(np.eye(len(intidxs))))
#        pom.mp.show()
#        self.ext_extinv = ssl.splu(MpP[extidxs, :][:, extidxs])
        
#        mp.subplot(1,3,1)
#        mp.spy(self.ext_allext, markersize=1)
#        mp.subplot(1,3,2)
#        mp.spy(self.int_allext, markersize=1)
#        mp.subplot(1,3,3)
#        mp.spy(self.ext_int, markersize=1)
#        mp.show()
        
        self.intsolveb = self.int_intinv.solve(b[intidxs].todense().A.squeeze())
        self.rhs = b[extidxs].todense().A.squeeze() - self.ext_int * self.intsolveb
        self.JMinv = None
        return [self.rhs]       
    
    @ppd.parallelmethod(None, ppd.tuplesum)
    def recoverinterior(self, xe):
        """ Recover all the local interior degrees from the exterior degrees
        
            returns a tuple of the contribution to the global solution vector and an indicator of what 
            degrees have been written to.  The indicator is used to average duplicate contributions for 
            overlapping methods.
        """ 
        x = np.zeros_like(self.intind, dtype=complex)
#        print len(self.intind), sum(self.intind), len(self.intminusint)
#        x[self.intind]
#        print len(self.int_ext * xe)
#        print len(self.int_intinv.solve(self.int_ext * xe))
#        print len(self.intsolveb)

        x[self.intind] = (self.intsolveb - self.int_intinv.solve(self.int_ext * xe[self.localext]))[self.intminusint]
        return x, self.intind*1
    
    @ppd.parallelmethod(None, ppd.tuplesum)
    def getData(self):
        return (self.S, self.P, self.G)

#
#@ppd.parallel()
#def test():
#    import time
#    time.sleep(np.random.rand() / 10)
#    return [ppd.comm.rank]
   
import pypwdg.parallel.main

def search():
    k = 2
    g = pcb.FourierHankel([-1,-1], [0], k)
    bdytag = "BDY"
    bnddata={bdytag:pcbd.dirichlet(g)}
    
    bounds=np.array([[0,1],[0,1]],dtype='d')
    npoints=np.array([200,200])
    gmresits = []
    params = []
    conds = []

    n = 6
    meshinfo = tum.regularsquaremeshinfo(n, bdytag)
    topology = pmm.Topology(meshinfo)
    partition = pmm.BespokePartition(meshinfo, topology, lambda n: np.arange(meshinfo.nelements).reshape(n, -1))    
    mesh = pmm.MeshView(meshinfo, topology, partition)
    
    mesh = pmo.overlappingPartitions(mesh)
    problem = psp.Problem(mesh,k,bnddata)
    
    npw = 7
    basisrule = pcb.planeWaveBases(2,k,npw)
    
    basisrule = pcbr.ReferenceBasisRule(pcbr.Dubiner(0))
    nquad = 4
           
        #    mesh = pmo.overlappingPartitions(pmo.overlappingPartitions(mesh))
            
           
        #    problem = psp.Problem(mesh, k, bnddata)
    dovolumes = True
           
    compinfo = psc.ComputationInfo(problem, basisrule, nquad)
    computation = psc.Computation(compinfo, pcp.HelmholtzSystem)
    sold = computation.solution(psc.DirectOperator(), psc.DirectSolver(), dovolumes=dovolumes)
    pom.output2dsoln(bounds, sold, npoints, show = False)
    for x in np.arange(-10,10,2):#, 1E-6, 1]:# 1j, -0.1, -0.1j]:
        for y in np.arange(-10,10,2):
            q = x + 1j * y
            perturbation = GeneralRobinPerturbation(compinfo, q)
        
            op = psd.GeneralSchwarzOperator(PerturbedSchwarzWorker(perturbation, mesh))
            callback = psi.ItCounter(100)
            solver = psi.GMRESSolver('ctor', callback)
            sol = computation.solution(op, solver, dovolumes=dovolumes)
            nn = len(op.rhs())
            M = np.hstack([op.multiply(xx).reshape(-1,1) for xx in np.eye(nn)])
            conds.append(np.linalg.cond(M))            
            params.append(q)
            gmresits.append(solver.callback.n)
        print conds
        print params
        print gmresits
        #sol = computation.solution(op, psi.BrutalSolver(np.complex))       

if __name__=="__main__":
#    search()
#    exit()
    k = 5
    n = 4
    g = pcb.FourierHankel([-1,-1], [0], k)
    bdytag = "BDY"
    bnddata={bdytag:pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1], g)}
    
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
    npw = 5
    basisrule = pcb.planeWaveBases(2,k,npw)
    dovolumes=False
    
#    basisrule = pcbr.ReferenceBasisRule(pcbr.Dubiner(0))
#    dovolumes = True
    nquad = 7
   
#    mesh = pmo.overlappingPartitions(pmo.overlappingPartitions(mesh))
    mesh = pmo.overlappingPartitions(mesh)
    
   
    problem = psp.Problem(mesh, k, bnddata)
    
    compinfo = psc.ComputationInfo(problem, basisrule, nquad)
    computation = psc.Computation(compinfo, pcp.HelmholtzSystem)
    sold = computation.solution(psc.DirectOperator(), psc.DirectSolver(), dovolumes=dovolumes)
    print "direct solution", sold.x
    pom.output2dsoln(bounds, sold, npoints, show = False)
    for p in [1]:#, 1, 1j, -0.1, -0.1j]:
        perturbation = GeneralRobinPerturbation(compinfo, p)
    
        w = PerturbedSchwarzWorker(perturbation, mesh)
        op = psd.GeneralSchwarzOperator(w)
#        op = psd.SchwarzOperator(mesh)
        
        sol = computation.solution(op, psi.GMRESSolver('ctor'), dovolumes=dovolumes)

        M, P, G = w.getData()
        sio.savemat('mpg.mat', {'M':M.tocsr(), 'P':P.tocsr(), 'G':G.tocsr().todense()})

        print "solution", sol.x

        xe = sol.x[op.extidxs]
        print "check jacobi multiply", np.max(np.abs(op.jacobimultiply(xe) - xe))
        print xe
        print op.jacobimultiply(xe)- xe


#        sol = computation.solution(op, psi.BrutalSolver(np.complex))       
#        print np.log10(ds / mds).astype(int).reshape(-1,npw)
        pom.output2dsoln(bounds, sol, npoints, show=False)
        n = len(op.rhs())
        M = np.hstack([op.multiply(x).reshape(-1,1) for x in np.eye(n)])
        sio.savemat('reduced.mat', {'R':M, 'b':op.rhs()})
        e = np.linalg.eigvals(M)
#        print e
        pom.mp.figure()
        pom.mp.scatter(e.real, e.imag)
        
        x = op.rhs()
        
        for _ in range(0):
            s = op.postprocess(x)
            pom.output2dsoln(bounds, psc.Solution(compinfo, s), npoints, show=False)
            for _ in range(10):
                x = op.jacobimultiply(x)
        
                 
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