'''
Import this module at the beginning of any module containing a main method to have worker processes spawned.
The workers get used by pypwdg.parallel.decorate

Created on Sep 12, 2010

@author: joel
'''
from pypwdg.parallel.mpiload import *

import os
import atexit
import sys
import time
import multiprocessing

if mpiloaded:
    # mpi things are happening.
    if comm.rank == 0:
        # when the boss goes home, the workers should too 
        def freeTheWorkers():
            # wake them up:
#            comm.bcast(root=0)
            # and send them home
            comm.scatter([(sys.exit, [], {})]*comm.size, root=0)
            
        atexit.register(freeTheWorkers)
    else:
        # we are a worker process
        # worker processes should contain their desire for lots of threads for BLAS 
        # this will only work if parallel is imported before anything that initialises the BLAS libraries
        # there is a better way to do this ... there's a C call that will set OMP_NUM_THREADS at run-time
        # todo: create a wrapper for it.   
        nt = max(multiprocessing.cpu_count() / comm.size, 1)
             
        os.putenv('OMP_NUM_THREADS', nt.__str__())
        while True:
            
            # For some unclear reason, the developers of openmpi think that it's acceptable for a thread to use 100% CPU
            # while waiting at a barrier.  I guess that for a symmetric algorithm with very small work packets, that might
            # be true, but our algorithm is not symmetric (master slave) and the work packets are not small (because
            # we don't expect miracles from mult-processing).  So it's vastly more efficient for our threads to poll
            # every 1ms to see if there's any work to do.    
#            mpi.world.irecv(source=0)
#            while(request.test() is None):
#                time.sleep(0.001)            
            
            task = comm.scatter(root=0)
            fn, args, kwargs = task
            res = fn(*args, **kwargs) 
            comm.gather(res, root=0)
#            mpi.gather(comm=mpi.world, value=res, root=0)
