'''
This is unused, or at least, not guaranteed to work.

Created on Mar 7, 2012

@author: joel
'''
import pypwdg.parallel.decorate as ppd
import pypwdg.utils.sparseutils as pus
import numpy as np
import scipy.sparse as ss
import pypwdg.mesh.mesh as pmm

@ppd.parallel()
def getCutFaces(mesh):
    ''' Extract the faces along which a partitioned mesh has been cut'''
    return map(tuple, list(mesh.faces[mesh.cutfaces==1]))

@ppd.distribute()
class SkeletonPartition(pmm.Partition):
    ''' A partition of a Skeleton mesh.  self.neighbourelts is updated to be the faces on the other side of the cut'''
    def __init__(self, meshview, skelemeshinfo, skelemeshtopology, skeleindicator, skeleindex):
        pmm.Partition.__init__(self, skelemeshinfo, skelemeshtopology, meshview.facepartition.diagonal()[skeleindicator].nonzero()[0])
        self.neighbourelts = skeleindex[np.array(meshview.connectivity.transpose() * skeleindicator, dtype=bool)]
#        print "SkeletonPartition", self.partition, self.neighbourelts
#        print skeleindicator, skeleindex, meshview.connectivity
    
class SkeletonisedDomain(object):
    ''' Information about a decomposed domain, along with its skeleton mesh.  '''
    def __init__(self, mesh, skeletontag):    
        meshinfo = mesh.basicinfo
        cutvertices = getCutFaces(mesh)
        
        boundaries = list(meshinfo.boundaries) # Get the boundaries from the underlying mesh
        boundaries.extend([(skeletontag, vs) for vs in cutvertices]) # And add a new boundary, labelled 'skeletontag'
        
        meshinfo2 = pmm.SimplicialMeshInfo(meshinfo.nodes, meshinfo.elements, meshinfo.elemIdentity, boundaries, meshinfo.dim) # Construct a new mesh, with the new boundary information
        self.mesh = pmm.MeshView(meshinfo2, pmm.Topology(meshinfo2), mesh.part)

        self.indicator = self.mesh.topology.faceentities==skeletontag #Identify the faces that are along the skeleton boundary
#        print self.indicator
        nskelelts = sum(self.indicator)
        self.index = np.ones(self.mesh.nfaces, dtype=int) * -1
        self.index[self.indicator] = np.arange(nskelelts) # self.index maps from mesh faces to the skeleton elements (contains -1 for non-skeleton faces)
        self.skeltomeshindex = self.indicator.nonzero()[0] # the indices of the faces in the mesh that are on the skeleton
        
        self.skel2mesh = pus.sparseindex(np.arange(nskelelts), self.skeltomeshindex, nskelelts, self.mesh.nfaces) # A sparse matrix that maps skeleton elements to mesh faces
        self.skel2oppmesh = self.skel2mesh * mesh.topology.connectivity # A sparse matrix that maps skeleton elements to the opposite mesh face
        self.skel2skel = self.skel2oppmesh * self.skel2mesh.transpose() # A sparse matrix that maps skeleton elements to the neighbouring skeleton element
#        print self.skel2skel
        
        skeletonelts = self.mesh.faces[self.indicator]
        meshinfo = pmm.SimplicialMeshInfo(self.mesh.nodes, skeletonelts, None, {}, self.mesh.dim -1)
        topology = pmm.Topology(meshinfo)
        partition = SkeletonPartition(mesh, meshinfo, topology, self.indicator, self.index)
        
        self.skeletonmesh =  pmm.MeshView(meshinfo, topology, partition) # Build the skeleton mesh.  N.B. it's topology should probably not be trusted.
#        print "skeltomesh", self.skeltomeshindex

    def expand(self, skeleeltarray):
        ''' Given data in an array corresponding to skeleton elements, create an array with the same data on the corresponding mesh faces '''
        meshfacearray = np.zeros(len(self.indicator), dtype=skeleeltarray.dtype)
        meshfacearray[self.indicator] = skeleeltarray
        return meshfacearray
    
            