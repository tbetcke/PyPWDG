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
import pypwdg.output.mploutput as pom
import matplotlib.pyplot as mp

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


def solvesquare(N, pdeg, g, basisrule, solve, k=20, dovolumes = True):
    impbd = pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g)
    bdytag = "BDY"    
    bnddata={bdytag:impbd}
    mesh = tum.regularsquaremesh(N, bdytag)
    p = 1 if pdeg < 1 else pdeg
    alpha = ((p*1.0)**2 * N)  / k
    beta = k / (p * 1.0*N) 
    problem=psp.Problem(mesh,k, bnddata)
    computation = psc.Computation(problem, basisrule, pcp.HelmholtzSystem, 15, alpha = alpha, beta = beta)
    solution = computation.solution(solve, dovolumes=dovolumes)
    return solution
#    
#
#def hConvergence(Ns, bdycond, basisrule, process, k = 20, scale = 4.0, pdeg = 1):
#    #bnddata={bdytag:pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g=g)}
#    bdytag = "BDY"    
#    bnddata={bdytag:bdycond}
#    for n in Ns:
#        mesh = tum.regularsquaremesh(n, bdytag)
#        if pdeg==0: pdeg = 1
#        alpha = ((pdeg*1.0)**2 * n)  / k
#        beta = k / (pdeg * 1.0*n) 
#        print alpha, beta
#        problem=psp.Problem(mesh,k, bnddata)
#        computation = psc.Computation(problem, basisrule, pcp.HelmholtzSystem, 15, alpha = alpha, beta = beta, delta = alpha)
#        solution = computation.solution(psc.DirectSolver().solve, dovolumes=True)
#        pom.output2dsoln()
#        process(n, solution)
        
def sourcesquareconvergence(Ns, Ps, NPWs, k = 20, polyonly = False):
    condfile = open("hankelcondition%s.txt"%k, 'a')
    fileroot="hankelsquare_k%s"%(k);
    bounds=np.array([[0,1],[0,1]],dtype='d')
    npoints=np.array([k * 10,k * 10], dtype=int)
    
    origin = np.array([-0.2,-0.2])
    
    g = pcb.FourierHankel(origin, [0], k)
    
    rt = PlaneWaveFromSourceRule(origin)
    condfile.write("Plane waves\n")
    for npw in NPWs:
        print "Number of plane waves ",npw
        pw = pcb.planeWaveBases(2, k, npw)
        fo = pof.ErrorFileOutput(fileroot + 'npw%s'%(npw), str(Ns), g, bounds, npoints)
        for N in Ns:
            solver = psc.DirectSolver(callback = lambda M: condfile.writelines("%s, %s, %s\n"%(npw, N, cond(M))))
            sol = solvesquare(N, 1, g, pw, solver.solve, k, dovolumes=False)
            fo.process(N, sol)

    if not polyonly:
        condfile.write("Polynomials + plane wave\n")    
        for pdeg in Ps:
            print "polynomial degree ",pdeg
            poly = pcbr.ReferenceBasisRule(pcbr.Dubiner(pdeg))                    
            polyrt = pcb.ProductBasisRule(poly, rt)
            fo = pof.ErrorFileOutput(fileroot + 'poly%srt'%(pdeg), str(Ns), g, bounds, npoints)
            for N in Ns:
                solver = psc.DirectSolver(callback = lambda M: condfile.write("%s, %s, %s\n"%(pdeg, N, cond(M))))
                sol = solvesquare(N, pdeg, g, polyrt, solver.solve, k)
                fo.process(N, sol)   
                
    if polyonly:
        condfile.write("Polynomials only\n")    
        for pdeg in Ps:
            print "polynomial degree ",pdeg
            poly = pcbr.ReferenceBasisRule(pcbr.Dubiner(pdeg))                    
            fo = pof.ErrorFileOutput(fileroot + 'poly%srt'%(pdeg), str(Ns), g, bounds, npoints)
            for N in Ns:
                solver = psc.DirectSolver(callback = lambda M: condfile.write("%s, %s, %s\n"%(pdeg, N, cond(M))))
                sol = solvesquare(N, pdeg, g, poly, solver.solve, k)
                fo.process(N, sol)   
        condfile.close()     
    


def examplepic(n, k=20):
    origin = np.array([-0.2,-0.2])
    bounds=np.array([[0,1],[0,1]],dtype='d')
    npoints=np.array([k * 20,k * 20], dtype=int)
    g = pcb.FourierHankel(origin, [0], k)
    pom.output2dfn(bounds, g.values, npoints, alpha=1.0, cmap=mp.cm.get_cmap('binary'))
    rt = PlaneWaveFromSourceRule(origin)
    mesh = tum.regularsquaremesh(n)
    pom.showmesh(mesh)
    etob = {}
    psp.Problem(mesh,k, None).populateBasis(etob, rt)
    pom.showdirections(mesh, etob)
    mp.show()
            
def ns(nmin, nmax, b):
    import math
    a = np.exp(np.arange(math.log(nmin), math.log(nmax), math.log(2)/b))
    return list(sorted(set(np.array(np.round(a), dtype=int))))

def cond(M):
    if np.prod(M.shape) < 4000000:
        c = np.linalg.cond(M.todense())
        print "Condition number = %s"%c
        return c
    return 0

#            pom.output2derror(bounds, sol, g.values, npoints, plotmesh=True)

