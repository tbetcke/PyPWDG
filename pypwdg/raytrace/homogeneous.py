'''
Created on Mar 8, 2011

@author: joel
'''

def trace(point, direction, mesh, reflectingbdys, nref=5):
    ''' Given a starting point and direction, trace a ray through a mesh.  
    
    reflectingbdys: entities that should reflect the ray.
    nref: number of reflections
    '''
    pass
    
class HomogenousTrace(object):
    ''' Knows how to trace through one element assuming homogenous material data''' 
    
    def __init__(self, mesh, reflectingbdys):
        self.mesh = mesh
        self.rbs = reflectingbdys
                
    def trace(self, face, direction):
        ''' Given an input face and a direction trace a path through the relevant element
        
        Returns: (e, f, d) where:
            e is the element that was traced through, 
            f is the next input face
            d is the next direction'''
        
        e = self.mesh.ftoe[face]
        fs = self.mesh.etof[e]
        
         
