'''
Created on Aug 15, 2010

@author: joel
'''

import pymeshpart.mesh

class MPIStructure(object):
    """ Partition the mesh across different MPI nodes ...
    
    Todo (amongst many): There's nothing clever in the partition yet, so going to be unnecessary vandermonde duplication
    """
    
    def __init__(self, mesh):
        facepartitions=None
        if mpi.rank==0:
            mesh.partition(mpi.size)
            facepartitions=mesh.facepartitions
            #partitions = [(p * mesh.nfaces) / mpi.size for p in range(0,mpi.size+1)]
            #facepartitions = [range(p0,p1) for p0,p1 in zip(partitions[:-1], partitions[1:])]
        self.facepartition = mpi.scatter(comm=mpi.world, values=facepartitions, root=0)
    
    def combine(self, M):
        return mpi.reduce(comm=mpi.world, value=M, op=lambda x,y: x + y, root=0)
    
def mpiloop(mulop, combineop = lambda x,y: x + y):
    if mpi.rank != 0:
        while True:
            x = mpi.broadcast(comm=mpi.world, value = None, root=0)
            if x is None: return None
            mpi.reduce(comm = mpi.world, value = mulop(x), op=combineop, root=0)
            
    else:
        def mainfn(x):
             mpi.broadcast(comm=mpi.world, value = x, root=0)
             if x is None: return None
             return mpi.reduce(comm = mpi.world, value = mulop(x), op=combineop, root=0) 
        
        return mainfn
        
