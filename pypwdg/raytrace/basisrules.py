'''
Created on May 14, 2011

@author: joel
'''
import pypwdg.core.bases.definitions as pcbb
import pypwdg.core.bases as pcb
import pypwdg.raytrace.wavefront as prw
import numpy as np
#import pypwdg.core.bases.utilities as pcbu

def normaliseandcompact(dirs):
    dirs = np.array(dirs)
    normeddirs =  dirs / np.sqrt(np.sum(dirs**2,axis=1)).reshape(-1,1)
    cdirs = []
    for d in normeddirs:
        newdir = True
        for cd in cdirs:
            if np.dot(cd,d) > 1 - 1E-2:
                newdir = False
                break
        if newdir: cdirs.append(d)
    return cdirs 

def etodsfromvtods(mesh, vtods):
    etods = []
    for vs in mesh.elements:
        etods.append(normaliseandcompact(sum([vtods[v] for v in vs],[])))
    return etods
    
def getetob(wavefronts, forwardidxs, mesh, bdys):
    vtods = prw.nodesToPhases(wavefronts, forwardidxs, mesh, bdys)
    etods = etodsfromvtods(mesh, vtods)
    etob = [[pcb.PlaneWaves(ds, k=10)] if len(ds) else [] for ds in etods]
    return etob




#
#class AugmentedBasisRule(object):
#    ''' Returns an augmented plane wave basis'''
#    def __init__(self, etods, augrule, nodes):
#        self.etods = etods
#        self.augrule = augrule
#        self.constant = pcbb.ConstantBasis()
#        self.nodes = nodes
#    
#    def populate(self, einfo):
#        ab = self.augrule.populate(einfo)[0] 
#        dirs, vs = self.etods[einfo.elementid]
#        bs = []
#        if len(dirs): bs.append(pcbb.PlaneWaves(dirs,einfo.k))
#        for v in vs:
#            bs.append(pcbb.FourierBessel(self.nodes[v], [-2,-1,0,1,2], einfo.k))
#            
#        if len(bs) == 0: bs.append(self.constant)
#        return bs
#        
#        return [pcbb.Product(pcbb.BasisCombine(bs), ab)]