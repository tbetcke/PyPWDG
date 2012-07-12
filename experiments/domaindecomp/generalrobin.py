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
#logging.getLogger('pypwdg.setup.indirect').setLevel(logging.DEBUG)
        
@ppd.distribute()
class SchwarzWorker(object):
    def __init__(self, mesh):
        self.mesh = mesh
        
    @ppd.parallelmethod()
    def calcneighbouridxs(self, system, sysargs, syskwargs):
        self.S,self.G = system.getSystem(*sysargs, **syskwargs) 
        return [self.S.subrows(self.mesh.neighbourelts)]
    
    @ppd.parallelmethod()
    def setexternalidxs(self, allextidxs):
#        print self.mesh.partition
        localidxs = self.S.subrows(self.mesh.partition)
        sl = set(localidxs)
        self.extidxs =  np.array(list(sl.intersection(allextidxs)), dtype=int)
        self.extidxs.sort()
        self.internalidxs = np.array(list(sl.difference(allextidxs)), dtype=int)

        log.info("local %s"%localidxs)
        log.info("external %s"%self.extidxs)
        log.info("internal %s"%self.internalidxs)

        M = self.S.tocsr()
        b = self.G.tocsr()
        self.n = b.shape[0]
        
#        print self.extidxs, allextidxs
        self.extrow = M[self.extidxs][:, allextidxs]
        Dint = M[self.internalidxs][:,self.internalidxs]
        print Dint.todense()
        self.DI = ssl.splu(Dint)
        Dext = M[self.extidxs][:, self.extidxs]
#        self.DE = ssl.splu(Dext)
        self.Cextint = M[self.internalidxs][:, self.extidxs]
        self.Cintext = M[self.extidxs][:, self.internalidxs]
        time.sleep(ppm.comm.rank * 0.1)
        
        self.localfromallext = np.zeros_like(self.extidxs)
        li = 0
        for gi, idx in enumerate(allextidxs):
            if idx == self.extidxs[li]:
                self.localfromallext[li] = gi
                li+=1
            if li==len(self.extidxs): break
        
#        self.DI.solve(np.zeros(Dint.shape[0]))
#        print type(np.zeros(Dint.shape[0]))
#        print type(b[self.internalidxs].todense().A)
        self.DIbint = self.DI.solve(b[self.internalidxs].todense().A.squeeze())
        rhs = b[self.extidxs].todense().A.squeeze() - self.Cintext * self.DIbint
        return [rhs]
        
        
    @ppd.parallelmethod()
    def multiplyext(self, x):
#        print self.extrow.shape, x.shape, self.Cintext.shape, self.Cextint.shape, x[self.localfromallext].shape
#        print "multiplyext", x
        y = self.extrow * x - self.Cintext * self.DI.solve(self.Cextint * x[self.localfromallext])
#        print x.shape, y.shape
#        print ppm.comm.rank, y
        return [y]  
    
    @ppd.parallelmethod()
    def precondext(self, x):
#        return x
        return [self.DE.solve(x[self.localfromallext])]
    
    @ppd.parallelmethod()
    def recoverfullsoln(self, ue):
        x = np.zeros(self.n, dtype=complex)
#        print x.shape, self.internalidxs.shape, self.DIbint.shape, self.DI.solve(self.Cextint * ue[self.localfromallext]).shape
        x[self.internalidxs] = self.DIbint - self.DI.solve(self.Cextint * ue[self.localfromallext])
        x[self.extidxs] = ue[self.localfromallext]
#        print x
        return x 
         
class SchwarzOperator(object):
    def __init__(self, mesh):
        self.workers = SchwarzWorker(mesh)
    
    def setup(self, system, sysargs, syskwargs):
        extidxs = np.unique(np.concatenate(self.workers.calcneighbouridxs(system, sysargs, syskwargs)))
        self.rhsvec = np.concatenate(self.workers.setexternalidxs(extidxs))
    
    def rhs(self):
        return self.rhsvec
    
    def multiply(self, x):
        y = np.concatenate(self.workers.multiplyext(x))
#        print y.shape
        return y
    
    def precond(self, x):
        return x
#        return np.concatenate(self.workers.precondext(x))
    
    def postprocess(self, x):
        return self.workers.recoverfullsoln(x)
    
import pypwdg.parallel.main

if __name__=="__main__":

    k = 5
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

    basisrule = pcb.planeWaveBases(2,k,5)
    nquad = 7
   
    problem = psp.Problem(mesh, k, bnddata)
    compinfo = psc.ComputationInfo(problem, basisrule, nquad)
    computation = psc.Computation(compinfo, pcp.HelmholtzSystem)
#    sol = computation.solution(SchwarzOperator(mesh), psi.GMRESSolver('ctor'))
    sol = computation.solution(SchwarzOperator(pmm.overlappingPartitions(mesh)), psi.GMRESSolver('ctor'))
    pom.output2dsoln(bounds, sol, npoints, show=True)