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

def parallelSum(x,y):
    if x is None: return y
    if y is None: return x
    return x + y
        

def parallel(scatterfn, reduceop = parallelSum):            
    def buildparallelwrapper(fn):
        # Check that there's mpi happening and that we're not already in a worker process 
        if mpi.world.size>1 and mpi.world.rank == 0:
            @functools.wraps(fn)
            def parallelwrapper(*arg, **kw):
                allargsaskw = dict(zip(inspect.getargspec(fn)[0], arg))
                allargsaskw.update(kw)
                
                scatterfnkw = dict([(key,val) for key, val in allargsaskw.items() if key in set(inspect.getargspec(scatterfn)[0])])
                scattereddata = scatterfn(mpi.world.size-1, **scatterfnkw)
                tasks = [None]                
                for sd in scattereddata:
                    taskkw = allargsaskw.copy()
                    taskkw.update(sd)
                    tasks.append((fn.__name__, fn.__module__, [], taskkw, reduceop))
#                print pickle.dumps(fn)
#                print tasks[-1][2].keys()
#                for k,v in tasks[-1][2].items(): print k, pickle.dumps(v)
#                print pickle.dumps(reduceop)
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