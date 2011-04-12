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
        
    def populate(self, mesh, einfo):
        return [pcbb.PlaneWaves(self.dirs, einfo.k * einfo.n)]        


class EntityNElementInfo(pcbu.ElementInfo):
    ''' Element info for variable n based on an entity-to-n map'''
    
    def __init__(self, mesh, k, entityton):
        pcbu.ElementInfo.__init__(self, mesh, k)
        self.entityton = entityton
    
    def n(self, e):
        entity = self.mesh.elemIdentity[e]
        return self.entityton[entity]

class FunctionNElementInfo(pcbu.ElementInfo):
    ''' Element info for variable n based on a function returning n given a point'''
    
    def __init__(self, mesh, k, fn):
        pcbu.ElementInfo.__init__(self, mesh, k)
        self.fn = fn
    
    def n(self, e):
        return self.fn(self.origin(e))