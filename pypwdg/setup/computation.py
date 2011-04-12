'''
Created on Apr 12, 2011

@author: joel
'''

import numpy
from pypwdg.core.vandermonde import LocalVandermondes
from pypwdg.core.physics import assemble
import pypwdg.core.bases as pcb

class Computation(object):
    """ Contains everything needed to solve a Problem  """
    def __init__(self, problem, elttobasis, usecache = True, dovols = False):
        self.problem = problem  
        self.elttobasis = elttobasis
        # Setup local Vandermondes        
        self.lv = LocalVandermondes(problem.mesh, elttobasis, problem.mqs, usecache=usecache)
        self.bndvs = []
        for data in problem.bnddata.values():
            bdyetob = pcb.constructBasis(problem.mesh, pcb.UniformBasisRule([data]))
            bndv = LocalVandermondes(problem.mesh, bdyetob, problem.mqs)        
            self.bndvs.append(bndv)
        stiffness, rhs = assemble(problem.mesh, problem.k, self.lv, self.bndvs, problem.mqs, self.elttobasis, problem.bnddata, problem.params, problem.emqs, dovols)
        self.stiffness = stiffness.tocsr()
        self.rhs = numpy.array(rhs.todense()).squeeze()
            
    def solve(self, solver="pardiso"):
        from pypwdg.setup.solution import Solution
        print "Solve linear system of equations"
        
        usepardiso = solver == "pardiso"
        useumfpack = solver == "umfpack"
        
        if not (usepardiso or useumfpack): raise Exception("Solver not known")
        
        if usepardiso:
            try:            
                from pymklpardiso.linsolve import solve
                (x, error) = solve(self.stiffness, self.rhs)
                if not error == 0: raise Exception("Pardiso Error")
            except ImportError:
                useumfpack = True
                
        if useumfpack:
            from scipy.sparse.linalg.dsolve.linsolve import spsolve as solve
            x = solve(self.stiffness, self.rhs)
        
        
        print "Relative residual: ", numpy.linalg.norm(self.stiffness * x - self.rhs) / numpy.linalg.norm(x)
        return Solution(self.problem, x, self.elttobasis, self.lv, self.bndvs)
                
