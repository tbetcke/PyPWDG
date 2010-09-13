'''
Created on Sep 12, 2010

@author: joel
'''

try:
    import boostmpi as mpi
except:    
    import boost.mpi as mpi

import inspect
import pickle
import functools
import numpy as np

def parallelSum(x,y):
    if x is None: return y
    if y is None: return x
    return x + y
        

def parallel(scatterfn, reduceop = parallelSum):     
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
                # broadcast a wake-up call to the worker threads.  
                mpi.broadcast(mpi.world, root=0)
                # munge all the arguments into key-word arguments
                allargsaskw = dict(zip(inspect.getargspec(fn)[0], arg))
                allargsaskw.update(kw)
                # work out which arguments to pass to scatterfn
                scatterfnkw = dict([(key,val) for key, val in allargsaskw.items() if key in set(inspect.getargspec(scatterfn)[0])])
                scattereddata = scatterfn(mpi.world.size-1, **scatterfnkw)
                tasks = [None] # task 0 goes to this process, which we want to remain idle.       
                # generate the arguments for the scattered functions         
                for sd in scattereddata:
                    taskkw = allargsaskw.copy()
                    taskkw.update(sd)
                    tasks.append((fn.__name__, fn.__module__, [], taskkw, reduceop))
                mpi.scatter(comm=mpi.world, values = tasks, root=0)
                ret = mpi.reduce(comm=mpi.world, value=None, op = reduceop, root = 0)
                return ret
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