'''
Created on Oct 31, 2011

@author: joel
'''
import numpy as np

msgs = [np.ones(10) * i for i in \
        range(5)]

import mpi4py.MPI as mpi
import time
comm = mpi.COMM_WORLD

time.sleep(1.0)

if comm.rank==0:
    msgs = [np.ones(10) * i for i in 
            range(comm.size)]
    comm.scatter(msgs)
else:
    msg = comm.scatter()
    print "My rank is %s"%(comm.rank), msg
    
