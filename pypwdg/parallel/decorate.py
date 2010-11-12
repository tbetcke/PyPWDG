'''
Created on Sep 12, 2010

@author: joel
'''

from pypwdg.parallel.mpiload import *

import functools
import pypwdg.parallel.proxy as ppp
import uuid
import sys
import inspect
import types
import operator

  
class methodwrapper(object):
    """ A pickable version of an instance method"""
    def __init__(self, m):
        self.methodname = m.__name__
    
    def __call__(self, *args, **kwargs):
        objself = args[0]
        return objself.__getattr__(self.methodname)(*args[1:], **kwargs)

class functionwrapper(object):
    """ A pickable version of a function
    
        Ordinary functions are picklable, but not if they've been replaced by a decorator
    """
    def __init__(self, f):
        self.module = f.__module__
        self.name = f.__name__
    
    def __call__(self, *args, **kwargs):
        __import__(self.module)
        return getattr(sys.modules[self.module],self.name)(*args, **kwargs)    

def wrapfn(fn):
    if type(fn) is types.FunctionType: return functionwrapper(fn)
    return fn
    
class Reducer:
    """ Need a class rather than a lambda fn because needs to be picklable"""
    def __init__(self,reduceop):
        self.reduceop = reduceop
    
    def __call__(self, x, y):
        if self.reduceop is None: return None
        if x is None: return y
        if y is None: return x
        return self.reduceop(x,y)

def scatterfncall(fn, args, reduceop=None):
    """ Scatter function calls to the worker processes
    
        fn: function to be scattered (must be picklable)
        args: list of (*args, *kwargs) pairs - same length as number of workers
        reduceop: reduction operator to apply to results (can be None)        
    """
    comm = mpi.COMM_WORLD

#    mpi.broadcast(mpi.world, root=0)
    
    tasks = [None] # task 0 goes to this process, which we want to remain idle.       
    # generate the arguments for the scattered functions         
    for a in args:
        tasks.append((fn, a[0], a[1]))
    comm.scatter(tasks, root=0)
    values = comm.gather(root = 0)
    ret = values if reduceop is None else reduce(reduceop, values[1:])
    return ret
             

def parallel(scatterargs, reduceop = operator.add):     
    """ A decorator that will parallelise a function (in some circumstances)
    
        The underlying function is run on all available worker processes.  The results are reduced back to
        the main process.  
        
        This is partially deprecated - it's probably nicer to distribute objects rather than functions (this 
        means that there's likely to be less communication as data is preserved in the workers) 
    
        scatterargs: a helper function that generates the arguments that should be sent to each process.
            the arguments to scatterargs should be the number of partitions, followed by the same arguments as
            the underlying function.  It should return a list of *arg **kwarg tuples
        reduceop: how to reduce the results
    """    
    
    def buildparallelwrapper(fn):
        # Check that there's mpi happening and that we're not already in a worker process 
        if mpiloaded and comm.rank==0:
            @functools.wraps(fn)
            def parallelwrapper(*arg, **kw):
                if scatterargs is None:
                    scatteredargs = [(arg,kw)]*(comm.size-1)
                else:
                    scatteredargs = scatterargs(comm.size-1)(*arg, **kw) 
                return scatterfncall(wrapfn(fn), scatteredargs, reduceop)
            return parallelwrapper
        else:
            return fn
    return buildparallelwrapper


def distribute(scatterargs=None):
    """ decorator that effectively distributes a class across worker processes.
    
        If we're in an mpi context then the __new__ method is replaced so that on the master process,
        a proxy object is created instead.  slave objects are created on each of the slave processes.
        When the proxy object is passed from the master to the slave, it will dynamically look up and
        reference the slave objects.  
        
        And methods marked as @parallelmethod will be distributed from the master to the slave.  
        Methods not marked as @parallelmethod will not be available on the master (there's no object for 
        them to act on).
        
        Why do we override __new__, rather than just doing a normal class decorator?  It's because
        decoration and pickling really don't get on very well.  We need to pickle the underlying 
        class for a proxy (in order to pickle the proxy itself).  If the decorator returns a different
        class object / function then this doesn't work.  Proxies could do the module / name munging
        themselves, but I think this is (just about) the less of two evils.        
    """

    def proxifynew(klass): 
        
        # Check that there's mpi happening and that we're not already in a worker process 
        if mpiloaded and comm.rank == 0:      
            @functools.wraps(klass)            
            def new(klass, *args, **kwargs):
                id = uuid.uuid4()
                if scatterargs is None:
                    scatteredargs = [((klass, id) + args,kwargs)]*(comm.size-1)
                else:
                    scatteredargs = [((klass, id) + s[0], s[1]) for s in scatterargs(comm.size-1)(*args, **kwargs)]

                scatterfncall(ppp.createproxy, scatteredargs)
                proxy = ppp.Proxy(klass, id)
                for name, m in inspect.getmembers(klass, inspect.ismethod):
                    pmdata = parallelmethods.get(m.im_func)
                    if pmdata is not None:
                        (mscatterargs, reduceop) = pmdata
                        def memberwrapper(*margs, **mkwargs):
                            if mscatterargs is None:
                                scatteredargs = [(margs,mkwargs)]*(comm.size-1)
                            else:
                                scatteredargs = mscatterargs(comm.size-1)(*margs, **mkwargs)
                            return scatterfncall(methodwrapper(m.im_func), scatteredargs, reduceop)
                        proxy.__setattr__(name, types.MethodType(memberwrapper, proxy, ppp.Proxy))
                            
                return proxy
            klass.__new__ = staticmethod(new)
        return klass
    return proxifynew


parallelmethods = {}

def parallelmethod(scatterargs = None, reduceop = operator.add):
    """ Decorator that marks an instance method for parallelisation"""
    def registermethod(fn): 
        if mpiloaded and comm.rank == 0:
            parallelmethods[fn] = (scatterargs, reduceop)
        return fn           
    return registermethod

def immutable(klass):
    """ Marking a klass as immutable allows us to serialise it a lot more efficiently 
    
        On instantiation, a copy of the object is placed in each process and wrapped by a Proxy   
    """
    
    if mpiloaded and comm.rank == 0:      
        @functools.wraps(klass)            
        def new(klass, *arg, **kw):
            obj = object.__new__(klass)
            obj.__init__(*arg, **kw)
            id = uuid.uuid4()
            scatterfncall(ppp.registerproxy, [((id, obj),{})] * (comm.size-1))
            return ppp.Proxy(klass, id, obj) 
        klass.__new__ = staticmethod(new)
    return klass

        
def partitionlist(numparts, l):
    """ Helper function that partitions a list into (almost) equal parts """
    np = len(l)
    return [l[np * i / numparts:np * (i+1)/numparts] for i in range(numparts)]

def tuplesum(x,y):
    return tuple(xk+yk for xk,yk in zip(x,y)) 

def npconcat(x,y):
    import numpy as np
    return np.concatenate((x,y))