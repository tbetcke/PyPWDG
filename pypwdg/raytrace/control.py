'''
Created on Apr 18, 2011

@author: joel
'''
import pypwdg.raytrace.element as pre
import pypwdg.raytrace.planewave as prp
import pypwdg.utils.quadrature as puq
import pypwdg.mesh.meshutils as pmmu

import numpy as np

def newdir(olddirs, d):
    for od in olddirs:
        if np.dot(od, d) > 1 - 1E-6: return False
    return True

def trace(face, point, direction, tracer, maxref=5, maxelts=-1):
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
        if newdir(eds, direction): eds.append(direction) 
        nelts-=1
        laste = e
    return etods

def tracefrombdy(problem, bdy, mqs, maxspace, dotrace):
    faces = problem.mesh.entityfaces[bdy]            
    for f in faces.tocsr().indices:            
        qp = mqs.quadpoints(f)
        qw = mqs.quadweights(f)
        thetas = prp.findpw(prp.L2Prod(problem.bnddata[bdy], (qp,qw), problem.k), 2, threshold = 0.2, maxtheta = 2)
        dirs = np.vstack([np.cos(thetas), np.sin(thetas)]).T
        n = problem.mesh.normals[f]
        ips = -np.dot(dirs, n)
        for dir, ip in zip(dirs, ips):
            if ip > 0 :
                intervals = np.ceil(ip / maxspace)+2
                refp = np.linspace(0,1,intervals)[1:-1]
                facedirs = problem.mesh.directions[f]
                facep = facedirs[0] + np.dot(refp.reshape(-1,1), facedirs[[1]])
                for p in facep:
                    dotrace(f, p, dir)

def tracefrombdy2(problem, bdy, inidirs, pointsperface, dotrace):
    faces = problem.mesh.entityfaces[bdy]            
    for f in faces.tocsr().indices:  
        refp = np.linspace(0,1,pointsperface+1)[1:-1]
        facedirs = problem.mesh.directions[f]
        facep = facedirs[0] + np.dot(refp.reshape(-1,1), facedirs[[1]])
        for p in facep:
            dir = inidirs(p).squeeze()
            dotrace(f, p, dir)
        
def tracemesh(problem, quadpoints, sources):
    etods = [[] for _ in range(problem.mesh.nelements)]
    tracer = pre.HomogenousTrace(problem.mesh, sources.keys())
    def dotrace(f, p, dir):
        for e, d1s in trace(f, p, dir, tracer).iteritems(): 
            d2s = etods[e]
            for d in d1s: 
                if newdir(d2s, d): d2s.append(d)
        
    for bdy, inidirs in sources.iteritems():
        tracefrombdy2(problem, bdy, inidirs, 3, dotrace)
    return etods

        
        
     