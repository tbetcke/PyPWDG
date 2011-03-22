'''
Created on Mar 22, 2011

@author: joel
'''

def initialrt(mesh, bdydata, direction, maxspace):
    """ Find starting points and directions for ray-tracing """
    
    for (bdy, bc) in bdydata.items():
        faces = mesh.entityfaces[bdy]            
        for f in faces.tocsr().indices:
            
            