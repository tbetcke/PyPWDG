'''
Created on Nov 11, 2010

@author: joel
'''
import boost.mpi as mpi
import numpy as np
from time import time

N = 1000000
#a = np.arange(N, dtype=float)
a = [np.arange(10) for i in range(N/10)]

if mpi.world.rank == 0:
    ts = []
    ts.append(time())
    mpi.broadcast(mpi.world, a, root=0)    
    ts.append(time())
    g = mpi.gather(mpi.world, a, root=0)
    ts.append(time())
    ta = np.array(ts)
    print "Times taken: ",(ta[1:] - ta[:-1])
    
else:
    mpi.broadcast(mpi.world, a, root=0)
    mpi.gather(mpi.world, a, root=0)