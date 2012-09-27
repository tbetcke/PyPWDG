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
    '''A Partition class that supports an overlapping method.  
    
        The neighbouring elements are added to each partition (so they are, strictly speaking, no longer
        a partition).  N.B. cutfaces is left the same
    '''
    def __init__(self, meshview):
        overlappart = np.sort(np.concatenate((meshview.partition, meshview.neighbourelts)))
        log.debug(overlappart)
        pmm.Partition.__init__(self, meshview.basicinfo, meshview.topology, overlappart, meshview.part.partidx)
        self.cutfaces = meshview.part.cutfaces
        self.oldneighbours = meshview.neighbourelts

def overlappingPartitions(meshview):
    return pmm.MeshView(meshview.basicinfo, meshview.topology, OverlappingPartition(meshview))