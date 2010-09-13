'''
Created on Sep 12, 2010

@author: joel
'''

try:
    import boostmpi as mpi
except:    
    import boost.mpi as mpi

def parallelSum(x,y):
    if x is None: return y
    if y is None: return x
    return x + y

def parallel(scatterfn, reduceop = parallelSum):            
    def buildparallelwrapper(fn):
        if mpi.world.size>=1 and mpi.world.rank == 0:
            # Check that there's mpi happening and that we're not already in a worker process     
            def parallelwrapper(*arg, **kw):
                argdata, kwdata = scatterfn(arg, kw)
                tasks = [None]                
                for j in range(1,mpi.world.size):
                    jarg = arg
                    jkw = kw
                    for (i, data) in argdata.items: jarg[i] = data[j]
                    for (key, data) in kwdata.items: jkw[key] = data[j]
                    tasks.append((fn, jarg, jkw, reduceop))
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