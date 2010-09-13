'''
Import this module at the beginning of any module containing a main method to have worker processes spawned.
The workers get used by pypwdg.parallel.decorate

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
import time

if mpi.world.size > 1:
    # mpi things are happening.
    if mpi.world.rank == 0:
        # when the boss goes home, the workers should too 
        def freeTheWorkers():
            # wake them up:
            mpi.broadcast(comm=mpi.world, root=0)
            # and send them home
            mpi.scatter(comm=mpi.world, values=[('exit','sys', [], {}, None)]*mpi.world.size, root=0)
            
        atexit.register(freeTheWorkers)
    else:
        # we are a worker process
        # worker processes should use single-threaded BLAS (the threading is at a higher level)
        # this will only work if parallel is imported before anything that initialises the BLAS libraries
        # there is a better way to do this ... there's a C call that will set OMP_NUM_THREADS at run-time
        # todo: create a wrapper for it.        
        os.putenv('OMP_NUM_THREADS', '1')
        while True:
            
            # For some unclear reason, the developers of openmpi think that it's acceptable for a thread to use 100% CPU
            # while waiting at a barrier.  I guess that for a symmetric algorithm with very small work packets, that might
            # be true, but our algorithm is not symmetric (master slave) and the work packets are not small (because
            # we don't expect miracles from mult-processing).  So it's vastly more efficient for our threads to poll
            # every 1ms to see if there's any work to do.    
            request = mpi.world.irecv(source=0)
            while(request.test() is None):
                time.sleep(0.001)            
            
            task = mpi.scatter(comm=mpi.world, values=None, root=0)
            fnname, fnmodule, args, kwargs, reduceop = task
            __import__(fnmodule)
            res = getattr(sys.modules[fnmodule],fnname)(*args, **kwargs)
            mpi.reduce(comm=mpi.world, value = res, op=reduceop, root=0)
