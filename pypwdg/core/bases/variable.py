'''
Classes to support the creation of a basis for problems with inhomogenous material data

Created on Apr 12, 2011

@author: joel
'''
import pypwdg.core.bases.definitions as pcbb        
import pypwdg.core.bases.utilities as pcbu    

class PlaneWaveVariableN(object):
    ''' Construct PlaneWave basis objects based on a variable n'''
    def __init__(self, dirs):
        self.dirs = dirs
        
    def populate(self, einfo):
        return [pcbb.PlaneWaves(self.dirs, einfo.kp(einfo.origin))]        


class EntityNElementInfo(pcbu.ElementInfo):
    ''' Element info for variable n based on an entity-to-n map
        entityton should be a dictionary of entities to either a scalar, or 
        a callable that returns a 1-D array given a set of points    
    '''    
    def __init__(self, mesh, k, entityton):
        pcbu.ElementInfo.__init__(self, mesh, k)
        self.entityton = entityton
                    
    def kp(self, e):
        entity = self.mesh.elemIdentity[e]
        n = self.entityton[entity]
        if callable(n): return lambda p: n(p) * self.k
        else: return lambda p: n * self.k