'''
Created on Apr 18, 2011

@author: joel
'''
import pypwdg.raytrace.element as pre
import pypwdg.parallel.decorate as ppd
import pypwdg.utils.timing as put
import numpy as np
import collections as cs

def newdir(olddirs, d):
    for od in olddirs:
        if np.dot(od, d) > 1 - 1E-6: return False
    return True

TracePoint = cs.namedtuple('TracePoint', 'face, point, direction')

def trace(tp, tracer, maxref=5, maxelts=-1):
    ''' Given a starting point on a face and direction, trace a ray through a mesh.  
    
    tracer: object with a trace method
    maxref: maximum number of reflections
    maxelts: maximum number of elements to trace
    '''
    etods = {}
    nrefs = maxref # number of remaining reflections allowed
    nelts = maxelts # number of remaining elets allowed
    laste = -1
    face, point, direction = tp
    while (nrefs !=0 and nelts !=0):
        nextinfo = tracer.trace(face, point, np.array(direction))
#        print nextinfo, nelts
        if nextinfo is None: break
        e, face, point, nextdir = nextinfo 
        if laste==e: nrefs-=1
        eds = etods.setdefault(e, [])
#       print eds
        if newdir(eds, direction): eds.append(direction) 
        nelts-=1
        laste = e
        direction = nextdir
    return etods

def etodcombine(etod1,etod2):
    for e,ds2 in etod2:
        ds1 = etod1.setdefault(e,[])
        for d in ds2:
            if newdir(ds1, d): ds1.append(d)
    return etod1

@ppd.parallel(lambda n: lambda problem, tracepoints, tracer: [((problem, tp, tracer),{}) for tp in ppd.partitionlist(n, tracepoints)],
              etodcombine)
def dotrace(problem, tracepoints, tracer):
    etods = [[] for _ in range(problem.mesh.nelements)]
    for tp in tracepoints:
        for e, d1s in trace(tp, tracer).iteritems(): 
            d2s = etods[e]
            for d in d1s: 
                if newdir(d2s, d): d2s.append(d)
                
    return etods

def getstartingtracepoints(problem, bdy, inidirs, pointsperface):
    faces = problem.mesh.entityfaces[bdy]   
    tracepoints = []     
    for f in faces.tocsr().indices:  
        refp = np.linspace(0,1,pointsperface+1)[1:-1]
        facedirs = problem.mesh.directions[f]
        facep = facedirs[0] + np.dot(refp.reshape(-1,1), facedirs[[1]])
        for p in facep:
            dir = inidirs(p).squeeze()
            tracepoints.append(TracePoint(f, p, dir))
    return tracepoints

@put.print_timing        
def tracemesh(problem, sources):
    tracer = pre.HomogenousTrace(problem.mesh, sources.keys())
    tracepoints = []
    for bdy, inidirs in sources.iteritems():
        tracepoints.extend(getstartingtracepoints(problem, bdy, inidirs, 3))
    
    etods = dotrace(problem, tracepoints, tracer)
    return dict([(e, np.array(ds).reshape(-1, problem.mesh.dim)) for e, ds in enumerate(etods)])

#def tracefrombdy(problem, bdy, mqs, maxspace, dotrace):
#    faces = problem.mesh.entityfaces[bdy]            
#    for f in faces.tocsr().indices:            
#        qp = mqs.quadpoints(f)
#        qw = mqs.quadweights(f)
#        thetas = pap.findpw(pap.L2Prod(problem.bnddata[bdy], (qp,qw), problem.k), 2, threshold = 0.2, maxtheta = 2)
#        dirs = np.vstack([np.cos(thetas), np.sin(thetas)]).T
#        n = problem.mesh.normals[f]
#        ips = -np.dot(dirs, n)
#        for dir, ip in zip(dirs, ips):
#            if ip > 0 :
#                intervals = np.ceil(ip / maxspace)+2
#                refp = np.linspace(0,1,intervals)[1:-1]
#                facedirs = problem.mesh.directions[f]
#                facep = facedirs[0] + np.dot(refp.reshape(-1,1), facedirs[[1]])
#                for p in facep:
#                    dotrace(f, p, dir)        
        
     