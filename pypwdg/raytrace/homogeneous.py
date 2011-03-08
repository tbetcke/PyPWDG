'''
Created on Mar 8, 2011

@author: joel
'''

def trace(point, direction, mesh, reflectingbdys, nref=5):
    ''' Given a starting point and direction, trace a ray through a mesh.  
    
    reflectingbdys: entities that should reflect the ray.
    nref: number of reflections
    '''
    
    