'''
Load MPI.

It might be nice for this module to interact with the logging in order to display the partition id in all log messages. 

Created on Nov 11, 2010

@author: joel
'''
import logging

mpiloaded = False
try:
    import mpi4py.MPI as mpi    
    comm = mpi.COMM_WORLD
    mpiloaded = comm.size > 1
except ImportError: 
    print "Failed to import mpi4py"
    logging.info("Failed to import mpi4py")


print "Using MPI:",mpiloaded
