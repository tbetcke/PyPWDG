'''
Created on Nov 11, 2010

@author: joel
'''
import mpi4py.MPI as mpi
import numpy as np
import cPickle
import cStringIO

from time import time



comm = mpi.COMM_WORLD
N = 200000
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
#    so = cStringIO.StringIO()
#    p = cPickle.Pickler(so, protocol=2)
#    p.dump(a)
#    ts.append(time())
#    print "bcast so"
#    comm.Bcast(so.getvalue(), root=0)
#    print "gather "
#    comm.gather(None, root=0)
#    ts.append(time())
    
    ta = np.array(ts)
    print "Times taken: ",(ta[1:] - ta[:-1])
    
else:
    x = comm.bcast(None, root = 0)
    comm.gather(a, root = 0)
    rbuf = np.zeros((N), dtype=float) 
    comm.Bcast(rbuf, root=0)
    comm.Gather(a, None, root=0)
#    si = cStringIO.StringIO()
#    comm.Bcast(si, root=0)
#    up = cPickle.Unpickler(si.getvalue())
#    aa = up.load()
#    print len(aa)
#    comm.gather(None,root=0)


if comm.rank==0:
    a = np.arange(50, dtype=complex)
    comm.Send([a, mpi.COMPLEX], 1)
elif comm.rank==1:
    a = np.empty(50, dtype=complex)
    comm.Recv([a, mpi.COMPLEX], 0)
    print a
