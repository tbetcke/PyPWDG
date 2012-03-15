'''
Created on Mar 7, 2012

@author: joel
'''
import pypwdg.parallel.decorate as ppd
import pypwdg.utils.sparseutils as pus
import numpy as np
import scipy.sparse as ss
import pypwdg.mesh.mesh as pmm

@ppd.distribute()
class SubMesh(pmm.EtofInfo):
    
    def __init__(self, mesh, internalbdytag):
        self.mesh = mesh

        pmm.EtofInfo.__init__(self, mesh.dim, len(mesh.partition), len(mesh.fs))

        localfaces = pus.sparseindex(np.arange(self.nfaces), mesh.fs, self.nfaces, mesh.nfaces)
        facemat = lambda m: localfaces * m * localfaces.transpose()
        
        innerbdyfaces = ss.spdiags(mesh.cutfaces[mesh.fs], [0], self.nfaces, self.nfaces)
                
        self.internal = facemat(mesh.internal) - innerbdyfaces
        self.boundary = facemat(mesh.boundary) + innerbdyfaces
        self.connectivity = facemat(mesh.connectivity)

        self.etof = mesh.etof[mesh.partition]
        
        self.entityfaces = dict([(e, facemat(m)) for (e,m) in mesh.entityfaces.items()]+[(internalbdytag, innerbdyfaces)])
        
        self.partition = np.arange(self.nelements)
        self.facepartition = ss.eye(self.nfaces, self.nfaces, dtype=int)
        
        self.directions = mesh.directions[mesh.fs]
        self.normals = mesh.normals[mesh.fs]
        self.dets = mesh.dets[mesh.fs]
        
        self.neighbourelts = np.array([])
        self.elements = np.array(mesh.elements)[mesh.partition]
        self.nodes = mesh.nodes

def skeletonmesh(mesh):
    return pmm.Mesh(mesh.nodes, mesh.faces[mesh.cutfaces], None, {}, mesh.dim - 1)
    
            