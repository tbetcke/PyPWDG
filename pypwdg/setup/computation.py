'''
Created on Apr 12, 2011

@author: joel
'''
import pypwdg.core.evaluation as pce
import pypwdg.core.errors as pces
import pypwdg.setup.problem as psp
import numpy as np

        
class DirectSolver(object):
    ''' Abstracts the choice of direct solver'''
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
        print "pardisosolve"
        (x, error) = solve(M,b.squeeze(),msglvl=0)
        if not error == 0: raise Exception("Pardiso Error")
        return x
    
    def umfpackSolve(self, M, b):
        from scipy.sparse.linalg.dsolve.linsolve import spsolve as solve
        print "umfpacksolve"
        return solve(M,b)
        
    def solve(self, system, sysargs, syskwargs):
        S,G = system.getSystem(*sysargs, **syskwargs)
        M = S.tocsr()
        print "Solve linear system of equations ",M.shape
        if np.prod(M.shape) < 500000:
            print "Norm %s"%np.linalg.norm(M.todense())
        b = np.array(G.todense()).squeeze()
        x = self.solvemethod(M,b).squeeze()
        print "Relative residual: ", np.linalg.norm(M * x -b) / np.linalg.norm(x)
        return x        

class Computation(object):
    ''' A class to manage the construction of a linear system for a Galerkin approximation to a problem and the computation of its solution
    
        problem: A problem object (must have a populateBasis method)
        basisrule: The rule to construct the basis
        systemklass: The class of the object that will construct the stiffness matrix and load vector (see p.c.p.HelmholtzSystem);
                    must have a getSystem method.
        args, kwargs: Additional parameters to pass to the system object
    '''
    def __init__(self, problem, basisrule, systemklass, *args, **kwargs):
        self.problem = problem
        self.basis = psp.constructBasis(problem, basisrule)        
        self.system = systemklass(problem, self.basis, *args, **kwargs)
                
    def solution(self, solve, *args, **kwargs):
        ''' Solve the system 
        
            solver: a function that takes a vbsr matrix S and a csr matrix G and returns S \ G
            args, kwargs: additional parameters to pass to the getSystem method
        '''
        x = solve(self.system, args, kwargs)
        return Solution(self.problem, self.basis, x)        

def noop(x): return x

class Solution(object):
    """ The solution to a Problem """
    def __init__(self, problem, basis, x):  
        self.mesh = problem.mesh
        self.basis = basis
        self.x = x
        self.problem=problem
        self.errors=None
        
    def getEvaluator(self, filter = noop):
        return pce.StructuredPointsEvaluator(self.mesh, self.basis, filter, self.x)
                 
    def evaluate(self, structuredpoints, returnPoints=False, filter = noop):
        vals, count = self.getEvaluator(filter).evaluate(structuredpoints)
        count[count==0] = 1
        if returnPoints:
            return vals / count, structuredpoints.toArray()
        else:
            return vals / count
     
    def computeErrors(self,quadpoints=10):
        d,n,bnd=pces.EvalElementError(self.mesh,quadpoints,self.basis,self.problem.bnddata).evaluate(self.x)
        self.errors={'ElemDirichlet': np.sqrt(d),
                    'ElemNeumann'  : 1.0/self.problem.k*np.sqrt(n),
                    'ElemBoundary' : np.sqrt(bnd),
                    'Dirichlet'    : np.sqrt(np.sum(d)),
                    'Neumann'      : 1.0/self.problem.k*np.sqrt(np.sum(n)),
                    'Boundary'     : np.sqrt(np.sum(bnd))}
        self.errors['ElemCombined']=np.sqrt(self.errors['ElemDirichlet']**2+self.errors['ElemNeumann']**2
                                    +self.errors['ElemBoundary']**2)
        self.errors['Combined']=np.sqrt(self.errors['Dirichlet']**2+self.errors['Neumann']**2
                                        +self.errors['Boundary']**2)
        
        
    def getError(self,key):
        """Return Error
        
           INPUT:
           key   - One of 'ElemDirichlet', 'ElemNeumann', 'ElemBoundary',
                   'Dirichlet', 'Neumann', 'Boundary'
        """
        
        if self.errors is None: self.computeErrors()
        return self.errors[key]
    
    
        
    
