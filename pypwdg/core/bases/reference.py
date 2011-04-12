'''
Classes used to implement finite elements based on a reference element

@author: joel
'''

import pypwdg.core.bases.definitions as pcbd
import pypwdg.utils.polynomial as pup

import numpy as np
 
class ReferenceBasisRule(object):
    """ A basis that uses a (affine) transformation to a reference element (old skool)"""
    def __init__(self, reference):
        self.reference = reference
        
    def populate(self, e):
        return [Reference(e.refmap, self.reference)]

class Reference(pcbd.Basis):
    
    def __init__(self, map, reference):
        self.mapi = map.inverse
        self.reference = reference
        self.n = reference.n
    
    def values(self, x):
        return self.reference.values(self.mapi.apply(x))
    
    def derivs(self, x, n = None):   
        derivs = np.dot(self.reference.derivs(self.mapi.apply(x)),self.mapi.linear.transpose())
        return derivs if n is None else np.sum(derivs.transpose([0,2,1]) * n[..., np.newaxis], axis = 1)  
    
class Dubiner(object):
    """ At some point, this should probably try to (pre-)cache results.  """
    
    def __init__(self, k):
        self.k = k
        self.n = ((k+1) * (k+2)) / 2
        
    def values(self, x):        
        return pup.DubinerTriangle(self.k, x).values()
    
    def derivs(self, x):
        return pup.DubinerTriangle(self.k, x).derivs().transpose([1,2,0])