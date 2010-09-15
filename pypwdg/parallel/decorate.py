'''
Created on Sep 12, 2010

@author: joel
'''

try:
    import boostmpi as mpi
except:    
    import boost.mpi as mpi

import inspect
import functools
import numpy as np
import pypwdg.parallel.proxy as ppp
import uuid
import sys
import inspect
import types


def parallelSum(x,y):
    if x is None: return y
    if y is None: return x
    return x + y
        

def scatterfncall(fn, args, reduceop):
    mpi.broadcast(mpi.world, root=0)
    
    tasks = [None] # task 0 goes to this process, which we want to remain idle.       
    # generate the arguments for the scattered functions         
    for a in args:
        tasks.append((fn, a[0], a[1], reduceop))
    mpi.scatter(comm=mpi.world, values = tasks, root=0)
    ret = mpi.reduce(comm=mpi.world, value=None, op = reduceop, root = 0)
    return ret
    
class methodwrapper(object):
    def __init__(self, klass, m):
        self.klass = klass
        self.methodname = m.__name__
    
    def __call__(self, *args, **kwargs):
        objself = args[0]
        return objself.__getattr__(self.methodname)(*args[1:], **kwargs)

class functionwrapper(object):
    def __init__(self, f):
        print "functionwrapper ",f
        self.module = f.__module__
        self.name = f.__name__
    
    def __call__(self, *args, **kwargs):
        __import__(self.module)
        return getattr(sys.modules[self.module],self.name)(*args, **kwargs)               

def wrapfn(fn):
    if type(fn) is types.FunctionType: return functionwrapper(fn)
    return fn


def parallel(scatterargs, reduceop = parallelSum):     
    """ A decorator that will parallelise a function (in some circumstances)
    
        The underlying function is run on all available worker processes.  The results are reduced back to
        the main process.   
    
        scatterfn: a helper function that performs the necessary changes to the arguments in order for the
            worker processes to each get a subset of the work.  scatterfn should return a list of dicts that 
            contain the replacement arguments.
        reduceop: how to reduce the results
    """    
    
    def buildparallelwrapper(fn):
        print "parallelwrapper ",fn
        # Check that there's mpi happening and that we're not already in a worker process 
        if mpi.world.size>1 and mpi.world.rank == 0:
            @functools.wraps(fn)
            def parallelwrapper(*arg, **kw):
                
                if scatterargs is None:
                    scatteredargs = [(arg,kw)]*(mpi.world.size-1)
                else:
                    scatteredargs = scatterargs(mpi.world.size-1)(*arg, **kw) 
                return scatterfncall(wrapfn(fn), scatteredargs, reduceop)
            return parallelwrapper
        else:
            return fn
    return buildparallelwrapper

def parallelTupleSum(x,y):
    if x is None: return y
    if y is None: return x
    return tuple(xx+ yy for (xx,yy) in zip(x,y))

def parallelNumpyConcatenate(x,y):
    if x is None: return y
    if y is None: return x
    return np.concatenate((x,y))
#
#def distribute(scatterargs):
#
#    def buildobjectcreator(klass): 
#        print "objectcreator ",klass  
#        
#        # Check that there's mpi happening and that we're not already in a worker process 
#        if mpi.world.size>1 and mpi.world.rank == 0:      
#            @functools.wraps(klass)            
#            def objectcreator(*args, **kwargs):
#                id = uuid.uuid4()
#                scatteredargs = [((klass.__module__, klass.__name__, id) + s[0], s[1]) for s in scatterargs(mpi.world.size-1)(*args, **kwargs)]
#                scatterfncall(ppp.createproxybyname, scatteredargs, parallelSum)
#                return ppp.Proxy(klass, id)
#            return objectcreator
#        else:
#            return klass
#    return buildobjectcreator

def distribute(scatterargs):

    def proxifynew(klass): 
        print "objectcreator ",klass  
        
        # Check that there's mpi happening and that we're not already in a worker process 
        if mpi.world.size>1 and mpi.world.rank == 0:      
            @functools.wraps(klass)            
            def new(klass, *args, **kwargs):
                id = uuid.uuid4()
                scatteredargs = [((klass, id) + s[0], s[1]) for s in scatterargs(mpi.world.size-1)(*args, **kwargs)]
                scatterfncall(ppp.createproxy, scatteredargs, parallelSum)
                proxy = ppp.Proxy(klass, id)
                for name, m in inspect.getmembers(klass, inspect.ismethod):
                    pmdata = parallelmethods.get(m.im_func)
                    if pmdata is not None:
                        print "Found ",m.im_func, m
                        (mscatterargs, reduceop) = pmdata
                        def memberwrapper(*margs, **mkwargs):
                            if mscatterargs is None:
                                scatteredargs = [(margs,mkwargs)]*(mpi.world.size-1)
                            else:
                                scatteredargs = mscatterargs(mpi.world.size-1)(*margs, **mkwargs)
                            return scatterfncall(methodwrapper(ppp.Proxy, m.im_func), scatteredargs, reduceop)
                        proxy.__setattr__(name, types.MethodType(memberwrapper, proxy, ppp.Proxy))
                            
                return proxy
            klass.__new__ = staticmethod(new)
        return klass
    return proxifynew


parallelmethods = {}

def parallelmethod(scatterargs, reduceop):
    def registermethod(fn): 
        if mpi.world.size > 1 and mpi.world.rank ==0:
            print "Registering ", fn
            parallelmethods[fn] = (scatterargs, reduceop)
        return fn           
    return registermethod

        
def partitionlist(numparts, l):
    np = len(l)
    return [l[np * i / numparts:np * (i+1)/numparts] for i in range(numparts)]

    