#__all__ = ["setup", "solve", "problem"]
#import computation
#from computation import Computation
#import problem
#from problem import Problem
#
#problem = None
#computation = None
#usecache = True
#
#def setup(mesh,k,nquadpoints,bnddata):
#    """ Convenience method to create global Problem object"""
#    global computation, problem
#    computation = None
#    problem = Problem(mesh,k,nquadpoints,bnddata)
#    return problem
#
#def computation(elttobasis):
#    """ Convenience method to create global Computation object"""
#    global computation, usecache
#    computation = Computation(problem, elttobasis, usecache)
#
#def solve(elttobasis = None, solver="pardiso"):
#    """ Convenience method to solve the global Problem"""
#    if elttobasis: computation(elttobasis)
#    return computation.solve(solver)    