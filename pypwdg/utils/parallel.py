'''
Created on Aug 15, 2010

@author: joel
'''
import boostmpi as mpi

class MPIStructure(object):
    """ Partition the mesh across different MPI nodes ...
    
    Todo (amongst many): There's nothing clever in the partition yet, so going to be unnecessary vandermonde duplication
    """
    
    def __init__(self, mesh):
        facepartitions = None
        if mpi.rank == 0:
            partitions = [(p * mesh.nfaces) / mpi.size for p in range(0,mpi.size+1)]
            facepartitions = [range(p0,p1) for p0,p1 in zip(partitions[:-1], partitions[1:])]
        self.facepartition = mpi.scatter(comm=mpi.world, values=facepartitions, root=0)
    
    def combine(self, M):
        return mpi.reduce(comm=mpi.world, value=M, op=lambda x,y: x + y, root=0)