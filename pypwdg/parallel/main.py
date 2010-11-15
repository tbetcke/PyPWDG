'''
Import this module at the beginning of any module containing a main method to have worker processes spawned.
The workers get used by pypwdg.parallel.decorate

Created on Sep 12, 2010

@author: joel
'''
from pypwdg.parallel.mpiload import  mpiloaded, comm

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
            from pypwdg.parallel.messaging import mastersend
            # wake them up:
#            comm.bcast(root=0)
            # and send them home
            mastersend([(sys.exit, [], {})]*comm.size)
            
        atexit.register(freeTheWorkers)
    else:
        # we are a worker process
        # worker processes should contain their desire for lots of threads for BLAS 
        # this will only work if parallel is imported before anything that initialises the BLAS libraries
        # there is a better way to do this ... there's a C call that will set OMP_NUM_THREADS at run-time
        # todo: create a wrapper for it.   
        nt = max(multiprocessing.cpu_count() / comm.size, 1)
             
        os.putenv('MKL_NUM_THREADS', nt.__str__())
        os.putenv('OMP_NUM_THREADS', nt.__str__())

def runParallel():
    from pypwdg.parallel.messaging import workerloop
    if mpiloaded and comm.rank!=0: workerloop()
