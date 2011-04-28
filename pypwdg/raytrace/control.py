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

def etodcombine(etod1,etod2):
    for ds1, ds2 in zip(etod1, etod2):
        for d in ds2:
            if newdir(ds1, d): ds1.append(d)
    return etod1

@ppd.distribute(lambda n: lambda problem, tracepoints, tracer: [((problem, tp, tracer),{}) for tp in ppd.partitionlist(n, tracepoints)],)
class RayTracing(object):
    
    def __init__(self, problem, tracepoints, tracer):
        self.etods = [[] for _ in range(problem.mesh.nelements)]
        self.reflections = []
        self.tracer = tracer
        for tp in tracepoints: self.trace(tp) 
    
    def adddir(self, e, dir):
        dirs = self.etods[e]
        if newdir(dirs, dir):
            dirs.append(dir)
            return True
        return False
    
    def trace(self, tp, maxref=5, maxelts=-1):
        ''' Given a starting point on a face and direction, trace a ray through a mesh.  
        
        tracer: object with a trace method
        maxref: maximum number of reflections
        maxelts: maximum number of elements to trace
        '''
        reflections = []
        nrefs = maxref # number of remaining reflections allowed
        nelts = maxelts # number of remaining elets allowed
        laste = -1
        face, point, direction = tp
        while (nrefs !=0 and nelts !=0):
            nextinfo = self.tracer.trace(face, point, np.array(direction))
    #        print nextinfo, nelts
            if nextinfo is None: break
            e, nextface, point, nextdir = nextinfo 
            if laste==e: # the ray was reflected 
                nrefs-=1
                reflections.append((face, direction))
                    
            self.adddir(e, direction) 
            nelts-=1
            laste = e
            face = nextface
            direction = nextdir
    
    @ppd.parallelmethod(None, etodcombine)
    def getDirections(self):
        return self.etods
    
    @ppd.parallelmethod()
    def getReflections(self):
        return self.reflections
        
    
def getstartingtracepoints(problem, bdy, inidirs, pointsperface):
    tracepoints = []     
    for f in problem.mesh.faceentities.nonzero()[0]:  
        refp = np.linspace(0,1,pointsperface+1)[1:-1]
        facedirs = problem.mesh.directions[f]
        facep = facedirs[0] + np.dot(refp.reshape(-1,1), facedirs[[1]])
        for p in facep:
            dir = inidirs(p).squeeze()
            tracepoints.append(TracePoint(f, p, dir))
    return tracepoints

def processreflections(problem, reflections):
    for f, d in reflections:
        
    
    

@put.print_timing        
def tracemesh(problem, sources):
    tracer = pre.HomogenousTrace(problem.mesh, sources.keys())
    tracepoints = []
    for bdy, inidirs in sources.iteritems():
        tracepoints.extend(getstartingtracepoints(problem, bdy, inidirs, 3))
    
    rt = RayTracing(problem, tracepoints, tracer)
    etods = rt.getDirections()
    
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
        
     