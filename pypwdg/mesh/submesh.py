'''
Created on Mar 7, 2012

@author: joel
'''
import pypwdg.parallel.decorate as ppd
import pypwdg.utils.sparseutils as pus
import numpy as np
import scipy.sparse as ss
import pypwdg.mesh.mesh as pmm

@ppd.parallel()
def getCutVertices(mesh):
    return list(mesh.faces[mesh.cutfaces==1].flatten())

@ppd.distribute()
class SkeletonPartition(pmm.Partition):
    def __init__(self, meshview, skelemeshinfo, skelemeshtopology, skeleindicator, skeleindex):
        pmm.Partition.__init__(self, skelemeshinfo, skelemeshtopology, meshview.facepartition.diagonal()[skeleindicator].nonzero()[0])
        self.neighbourelts = skeleindex[np.array(meshview.connectivity * skeleindicator, dtype=bool)]
        
#        
#class SkeletonFaceMap(object):
#    
#    def __init__(self, mesh, skeletontag):
##        print "SkeletonFaceMap.init"
#        self.indicator = mesh.topology.faceentities==skeletontag
#        self.index = np.ones(mesh.nfaces, dtype=int) * -1
#        self.index[self.indicator] = np.arange(sum(self.indicator))
#        self.skeltomeshindex = self.indicator.nonzero()[0]
#        self.mesh = mesh
#        
#    def expand(self, skeleeltarray):
#        meshfacearray = np.zeros(len(self.indicator), dtype=skeleeltarray.dtype)
#        meshfacearray[self.indicator] = skeleeltarray
#        return meshfacearray
#    
#    def partition(self):
#        return self.mesh.facepartition.diagonal()[self.indicator].nonzero()[0]
#
#
#def skeletonMesh(mesh, skeletontag):
#    ''' Creates a codimension 1 submesh consisting of the faces associated with the boundary highlighted by skeletontag
#        Also returns the element->face map from the submesh back to the original mesh
#    '''
#    skeletonfacemap = SkeletonFaceMap(mesh, skeletontag)
#    skeletonelts = mesh.faces[skeletonfacemap.indicator]
#    meshinfo = pmm.SimplicialMeshInfo(mesh.nodes, skeletonelts, None, {}, mesh.dim -1)
#    topology = pmm.Topology(meshinfo)
#    partition = SkeletonPartition(mesh, meshinfo, topology, skeletonfacemap)
#    return pmm.MeshView(meshinfo, topology, partition), skeletonfacemap
#    
#    
    
class SkeletonisedDomain(object):
    def __init__(self, meshinfo, skeletontag):    
        self.underlyingmesh = pmm.meshFromInfo(meshinfo)
        cutvertices = getCutVertices(self.underlyingmesh)
        
        boundaries = list(meshinfo.boundaries)
        boundaries.append((skeletontag, cutvertices))
        
        meshinfo2 = pmm.SimplicialMeshInfo(meshinfo.nodes, meshinfo.elements, meshinfo.elemIdentity, boundaries, meshinfo.dim)
        self.mesh = pmm.MeshView(meshinfo2, pmm.Topology(meshinfo2), self.underlyingmesh.part)

        self.indicator = self.mesh.topology.faceentities==skeletontag
#        print self.indicator
        nskelelts = sum(self.indicator)
        self.index = np.ones(self.mesh.nfaces, dtype=int) * -1
        self.index[self.indicator] = np.arange(nskelelts)
        self.skeltomeshindex = self.indicator.nonzero()[0]
        
        self.skel2mesh = pus.sparseindex(np.arange(nskelelts), self.skeltomeshindex, nskelelts, self.mesh.nfaces)
        self.skel2oppmesh = self.skel2mesh * self.underlyingmesh.topology.connectivity
        self.skel2skel = self.skel2mesh * self.underlyingmesh.connectivity * self.skel2mesh.transpose()
        
        skeletonelts = self.mesh.faces[self.indicator]
        meshinfo = pmm.SimplicialMeshInfo(self.mesh.nodes, skeletonelts, None, {}, self.mesh.dim -1)
        topology = pmm.Topology(meshinfo)
        partition = SkeletonPartition(self.underlyingmesh, meshinfo, topology, self.indicator, self.index)
        
        self.skeletonmesh =  pmm.MeshView(meshinfo, topology, partition)

    def expand(self, skeleeltarray):
        meshfacearray = np.zeros(len(self.indicator), dtype=skeleeltarray.dtype)
        meshfacearray[self.indicator] = skeleeltarray
        return meshfacearray
    
            