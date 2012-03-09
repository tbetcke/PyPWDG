'''
Created on Mar 7, 2012

@author: joel
'''
import pypwdg.parallel.decorate as ppd
import pypwdg.utils.sparseutils as pus
import numpy as np
import scipy.sparse as ss
import pypwdg.mesh.mesh as pmm

@ppd.distribute
class SubMesh(pmm.EtofInfo):
    
    def __init__(self, mesh, internalbdytag):
        self.mesh = mesh
        localfaces = pus.sparseindex(np.arange(mesh.fs), mesh.fs, len(mesh.fs), mesh.nfaces)
        facemat = lambda m: localfaces * m * localfaces.transpose()

        pmm.EtofInfo.__init__(mesh.dim, len(mesh.partition), len(mesh.fs))
        
        innerbdyfaces = facemat(mesh.partition.cutfaces)
        
        self.internal = facemat(mesh.internal) - innerbdyfaces
        self.boundary = facemat(mesh.boundary) + innerbdyfaces
        self.connectivity = facemat(mesh.connectivity)

        self.etof = mesh.etof[mesh.partition]
        
        self.entityfaces = dict([(e, self.facemat(m)) for (e,m) in mesh.entityfaces.items()]+[(internalbdytag, innerbdyfaces)])
        
        self.partition = np.arange(self.nelements)
        self.facepartition = np.arange(self.nfaces)
        
        self.directions = mesh.directions[mesh.fs]
        self.normals = mesh.normals[mesh.fs]
        self.dets = mesh.dets[mesh.fs]
        

        