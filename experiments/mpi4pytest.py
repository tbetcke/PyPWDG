'''
Created on Nov 11, 2010

@author: joel
'''
import mpi4py.MPI as mpi
import numpy as np
from time import time

comm = mpi.COMM_WORLD
N = 20000000
a = np.arange(N, dtype=float)
#a = [np.arange(10) for i in range(N/10)]

if comm.rank == 0:
    ts = []
    ts.append(time())
    comm.bcast(a, root = 0)
    ts.append(time())
    g = comm.gather(a, root = 0)
    ts.append(time())
    comm.Bcast(a, root=0)
    ts.append(time())
    rbuf = np.zeros((N*comm.size), dtype=float) 
    comm.Gather(a, rbuf, root=0)
    ts.append(time())
    ta = np.array(ts)
    print "Times taken: ",(ta[1:] - ta[:-1])
    
else:
    x = comm.bcast(None, root = 0)
    comm.gather(a, root = 0)
    rbuf = np.zeros((N), dtype=float) 
    comm.Bcast(rbuf, root=0)
    comm.Gather(a, None, root=0)
