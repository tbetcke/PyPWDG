'''
Created on Sep 12, 2010

@author: joel
'''

try:
    import boostmpi as mpi
except:    
    import boost.mpi as mpi

import os
import atexit
import sys

if mpi.world.size > 1:
    # mpi things are happening.
    if mpi.world.rank == 0:
        # when the boss goes home, the workers should too 
        atexit.register(mpi.scatter, comm=mpi.world, values=[('exit','sys', [], {}, None)]*mpi.world.size, root=0)
    else:
        # we are a worker process
        # worker processes should use single-threaded BLAS (the threading is at a higher level)
        # this will only work if parallel is imported before anything that initialises the BLAS libraries
        # there is a better way to do this ... there's a C call that will set OMP_NUM_THREADS at run-time
        # todo: create a wrapper for it.        
        os.putenv('OMP_NUM_THREADS', '1')
        while True:
            print "Pre-scatter"
            task = mpi.scatter(comm=mpi.world, values=None, root=0)
            print "Post-scatter"
            fnname, fnmodule, args, kwargs, reduceop = task
            print fnmodule, fnname
            __import__(fnmodule)
            res = sys.modules[fnmodule].__getattribute__(fnname)(*args, **kwargs)
            mpi.reduce(comm=mpi.world, value = res, op=reduceop, root=0)
