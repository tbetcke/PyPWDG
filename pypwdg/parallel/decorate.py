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

def parallelSum(x,y):
    if x is None: return y
    if y is None: return x
    return x + y
        

def scatterfncall(fn, args, reduceop):
    mpi.broadcast(mpi.world, root=0)
    
    tasks = [None] # task 0 goes to this process, which we want to remain idle.       
    # generate the arguments for the scattered functions         
    for a in args:
        tasks.append((fn.__name__, fn.__module__, a[0], a[1], reduceop))
    mpi.scatter(comm=mpi.world, values = tasks, root=0)
    ret = mpi.reduce(comm=mpi.world, value=None, op = reduceop, root = 0)
    return ret
    

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
        # Check that there's mpi happening and that we're not already in a worker process 
        if mpi.world.size>1 and mpi.world.rank == 0:
            # this means that we can wrap a print_timing around the class and still get sensible results
            @functools.wraps(fn)
            def parallelwrapper(*arg, **kw):
                scatteredargs = scatterargs(mpi.world.size-1)(*arg, **kw)  
                return scatterfncall(fn, scatteredargs, reduceop)
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

def distribute(scatterargs):
    def buildobjectcreator(klass):    
        # Check that there's mpi happening and that we're not already in a worker process 
        if mpi.world.size>1 and mpi.world.rank == 0:  
            @functools.wraps(klass)                 
            def objectcreator(*args, **kwargs):
                print args
                id = uuid.uuid4()
                scatteredargs = [((klass, id) + s[0], s[1]) for s in scatterargs(mpi.world.size-1)(*args, **kwargs)]
                scatterfncall(ppp.createproxy, scatteredargs, parallelSum)
                return ppp.Proxy(klass, id)
            return objectcreator
        else:
            return klass
    return buildobjectcreator

        
def partitionlist(numparts, l):
    np = len(l)
    return [l[np * i / numparts:np * (i+1)/numparts] for i in range(numparts)]

    