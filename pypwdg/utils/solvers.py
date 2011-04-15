'''
Created on Apr 15, 2011

@author: joel
'''
import numpy as np
        
class DirectSolver(object):
    
    def __init__(self, solver="best"):
        
        solvers = {'best':self.bestSolve, 'pardiso':self.pardisoSolve, 'umfpack': self.umfpackSolve}
        try:
            self.solvemethod = solvers[solver]
        except KeyError:
            raise Exception("Solver %s not known"%(solver))
    
    def bestSolve(self, M, b):
        try:
            return self.pardisoSolve(M,b)
        except ImportError:
            return self.umfpackSolve(M, b)
    
    def pardisoSolve(self, M, b):
        from pymklpardiso.linsolve import solve
        (x, error) = solve(M,b)
        if not error == 0: raise Exception("Pardiso Error")
        return x
    
    def umfpackSolve(self, M, b):
        from scipy.sparse.linalg.dsolve.linsolve import spsolve as solve
        return solve(M,b)
        
    def solve(self, M, b):
        print "Solve linear system of equations"
        x = self.solvemethod(M,b).squeeze()
        print "Relative residual: ", np.linalg.norm(M * x -b) / np.linalg.norm(x)
        return x
        
        #return Solution(self.problem, x, computation.elttobasis, computation.facevandermondes, self.bdyvandermondes)
 