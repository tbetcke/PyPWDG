'''
Created on Apr 18, 2011

@author: joel
'''
import numpy as np

def trace(point, direction, face, tracer, maxref=5, maxelts=-1):
    ''' Given a starting point on a face and direction, trace a ray through a mesh.  
    
    tracer: object with a trace method
    maxref: maximum number of reflections
    maxelts: maximum number of elements to trace
    '''
    etods = {}
    nrefs = maxref # number of remaining reflections allowed
    nelts = maxelts # number of remaining elets allowed
    laste = -1
    while (nrefs !=0 and nelts !=0):
        nextinfo = tracer.trace(face, point, np.array(direction))
#        print nextinfo, nelts
        if nextinfo is None: break
        e, face, point, direction = nextinfo 
        if laste==e: nrefs-=1
        eds = etods.setdefault(e, [])
        eds.append(direction)
        nelts-=1
        laste = e
    return etods

class RayTracer(object):
    def __init__(self, problem):
        pass
     