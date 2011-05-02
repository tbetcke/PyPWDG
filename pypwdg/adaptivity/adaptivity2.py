'''
Created on 1 Nov 2010

@author: joel
'''
import pypwdg.parallel.decorate as ppd
import pypwdg.parallel.distributeddict as ppdd
import pypwdg.core.bases as pcb
import pypwdg.setup.computation as psc
import pypwdg.setup.problem as psp
import pypwdg.adaptivity.planewave as pap

import pypwdg.utils.quadrature as puq
import pypwdg.mesh.meshutils as pmmu

import numpy as np


@ppd.distribute()
class BasisController(object):
    
    def __init__(self, mesh, quadpoints, etods, k, nfb):
        self.k = k
        self.mesh = mesh
        self.etods = etods
        self.nfb= nfb
        _, equad = puq.quadrules(mesh.dim, quadpoints)
        self.mqs = pmmu.MeshElementQuadratures(mesh, equad)
        self.fbrule = pcb.FourierBesselBasisRule(np.arange(0,self.nfb) - self.nfb/2)
    
    @ppd.parallelmethod(None, None)
    def populate(self, einfo):
        dirs = self.etods[einfo.elementid]
#        print dirs
        pw = [pcb.PlaneWaves(dirs,einfo.k)] if len(dirs) else []
        fb = self.fbrule.populate(einfo)
        return pw + fb
            
    @ppd.parallelmethod(None, None)
    def adapt(self, basis, x):
        for e in self.mesh.es:            
            def g(p):
#                print basis.getValues(e,p).shape, x[basis.indices[e]:basis.indices[e] + basis.sizes[e]].shape
                return np.dot(basis.getValues(e, p), x[basis.indices[e]:basis.indices[e] + basis.sizes[e]])            
            ips = pap.L2Prod(g, (self.mqs.quadpoints(e), self.mqs.quadweights(e)), self.k)
            dirs = pap.findpwds(ips, diameter = 2, threshold = 0.05, maxtheta = 5)
            self.etods[e] = dirs.transpose() 

class AdaptiveComputation(object):
    
    def __init__(self, problem, controller, systemklass, *args, **kwargs):
        self.problem = problem
        self.etobmanager = ppdd.ddictmanager(ppdd.elementddictinfo(problem.mesh, True), True)
        self.etob = self.etobmanager.getDict()
        self.controller = controller
        self.createsys = lambda basis: systemklass(problem, basis, *args, **kwargs)
    
    def solve(self, solve, nits, output = None, *args, **kwargs):
        for i in range(nits):
            psp.localPopulateBasis(self.etob, self.controller, self.problem)
            self.etobmanager.sync()   
            basis = pcb.ElementToBases(self.etob, self.problem.mesh)
            print "Basis size: ",basis.getIndices()[-1]
            print "Average basis size: ", basis.getIndices()[-1]*1.0 / self.problem.mesh.nelements
            system = self.createsys(basis)
            x = solve(system, args, kwargs)
            solution = psc.Solution(self.problem, basis, x)  
            if output: output(i, solution)
            if i == nits-1: break
            self.controller.adapt(basis, solution.x)
    
        return solution
            
        