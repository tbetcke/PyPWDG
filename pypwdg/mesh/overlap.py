'''
Created on Jul 23, 2012

@author: joel
'''
import numpy as np
import pypwdg.parallel.decorate as ppd
import pypwdg.mesh.mesh as pmm
import logging
log = logging.getLogger(__name__)

@ppd.distribute()
class OverlappingPartition(pmm.Partition):
    '''Add in the neighbouring elements, but keep the cutfaces'''
    def __init__(self, meshview):
        overlappart = np.sort(np.concatenate((meshview.partition, meshview.neighbourelts)))
        log.debug(overlappart)
        pmm.Partition.__init__(self, meshview.basicinfo, meshview.topology, overlappart, meshview.part.partidx)
        self.cutfaces = meshview.part.cutfaces

def overlappingPartitions(meshview):
    return pmm.MeshView(meshview.basicinfo, meshview.topology, OverlappingPartition(meshview))