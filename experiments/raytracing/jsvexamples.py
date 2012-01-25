'''
Created on Jan 25, 2012

@author: joel
'''
import pypwdg.test.utils.mesh as tum
import pypwdg.setup.problem as psp
import pypwdg.setup.computation as psc
import pypwdg.core.physics as pcp
import pypwdg.core.bases as pcb
import pypwdg.core.boundary_data  as pcbd
import pypwdg.core.bases.reference as pcbr
import pypwdg.output.file as pof

import numpy as np

class PlaneWaveFromSourceRule(object):
    
    def __init__(self, sourceorigin):
        self.sourceorigin=sourceorigin
        
    def populate(self, einfo):
        import pypwdg.core.bases.definitions as pcbb
        import numpy.linalg as nl
        offset = einfo.origin - self.sourceorigin 
        direction = offset / nl.norm(offset) 
        return [pcbb.PlaneWaves([direction],einfo.k)]


def hConvergence(Ns, bdycond, basisrule, process, k = 20, scale = 4.0, pdeg = 1):
    #bnddata={bdytag:pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g=g)}
    bdytag = "BDY"    
    bnddata={bdytag:bdycond}
    for n in Ns:
        mesh = tum.regularsquaremesh(n, bdytag)
        if pdeg==0: pdeg = 1
        alpha = ((pdeg*1.0)**2 * n)  / k
        beta = k / (pdeg * 1.0*n) 
        print alpha, beta
        problem=psp.Problem(mesh,k, bnddata)
        computation = psc.Computation(problem, basisrule, pcp.HelmholtzSystem, 15, alpha = alpha, beta = beta)
        solution = computation.solution(psc.DirectSolver().solve, dovolumes=True)
        process(n, solution)
        
def sourcesquareconvergence(maxN, maxP, k = 20):
    fileroot="hankelsquare_k%s"%(k);
    bounds=np.array([[0,1],[0,1]],dtype='d')
    npoints=np.array([k * 10,k * 10], dtype=int)
    
    origin = np.array([-2,3])
    
    g = pcb.FourierHankel(origin, [0], k)
    impbd = pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g)

    rt = PlaneWaveFromSourceRule(origin)
    Ns = range(3, maxN)
    for pdeg in range(maxP+1):
        print "polynomial degree ",pdeg
        poly = pcbr.ReferenceBasisRule(pcbr.Dubiner(pdeg))                    
        polyrt = pcb.ProductBasisRule(poly, rt)
        fo = pof.ErrorFileOutput(fileroot + 'poly%srt'%(pdeg), str(Ns), g, bounds, npoints)
        hConvergence(Ns, impbd, polyrt, fo.process, k, pdeg)        

