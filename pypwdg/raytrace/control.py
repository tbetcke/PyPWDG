'''
Created on Apr 18, 2011

@author: joel
'''
import pypwdg.raytrace.element as pre
import pypwdg.parallel.decorate as ppd
import pypwdg.utils.timing as put
import pypwdg.core.bases.utilities as pcbu
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

@ppd.distribute(lambda n: lambda mesh, tracepoints, tracer: [((mesh, tp, tracer),{}) for tp in ppd.partitionlist(n, tracepoints)],)
class RayTracing(object):
    
    def __init__(self, mesh, tracepoints, tracer, maxref = 5, maxelts = -1):
        self.etods = [[] for _ in range(mesh.nelements)]
        self.reflections = []
        self.tracer = tracer
        for tp in tracepoints: self.trace(tp,maxref, maxelts) 
    
    def adddir(self, e, dir):
        dirs = self.etods[e]
        if newdir(dirs, dir):
            dirs.append(dir)
            return True
        return False
    
    def trace(self, tp, maxref, maxelts):
        ''' Given a starting point on a face and direction, trace a ray through a mesh.  
        
        tracer: object with a trace method
        maxref: maximum number of reflections
        maxelts: maximum number of elements to trace
        '''
#        print "trace ", tp
        nrefs = maxref # number of remaining reflections allowed
        nelts = maxelts # number of remaining elements allowed
        laste = -1
        face, point, direction = tp
        while (nrefs !=0 and nelts !=0):
            nextinfo = self.tracer.trace(face, point, np.array(direction))
#            print nextinfo, nelts
            if nextinfo is None: break
            e, nextface, point, nextdir = nextinfo 
            if laste==e: # the ray was reflected 
                nrefs-=1
                self.reflections.append((face, direction))
                    
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
    for f in (problem.mesh.faceentities==bdy).nonzero()[0]:  
        refp = np.linspace(0,1,pointsperface+1)[1:-1]
        facedirs = problem.mesh.directions[f]
        facep = facedirs[0] + np.dot(refp.reshape(-1,1), facedirs[[1]])
        for p in facep:
            dir = inidirs(p).squeeze()
            tracepoints.append(TracePoint(f, p, dir))
    return tracepoints

def processreflections(problem, reflections):
    ''' Takes a list of reflections and returns a new set of TracePoints giving initial ray tracing parameters
        for the diffraction
    '''
    diffractedpoints = [] # this will be a list of initial rays coming from diffraction
    vertices = set() # keep track of the vertices that we've visited
    vtof = problem.mesh.vtof * problem.mesh._boundary # sparse indicator matrix mapping vertices to boundary faces
    vtof.eliminate_zeros()
    dim = problem.mesh.dim
    offset = 1E-3 # we can't raytrace from a vertex, so this is how much we'll move inside each face from the verte
    dirs = pcbu.uniformdirs(dim, 24) # We're going to trace N uniform rays from each vertex (for 2D)
    for f, d in reflections: # We've been given a list of previous reflections as (face, direction) pairs
        for v in problem.mesh.faces[f]: # what vertices are associated with this face?
            if v not in vertices: # If we haven't already considered this vertex (N.B. this could have been done with vtof)
                vp = problem.mesh.nodes[v] # The physical points for the vertex
                vfs = vtof.getrow(v).indices # The neighbouring boundary faces for the vertex
                ns = problem.mesh.normals[vfs] 
                if np.abs(np.linalg.det(ns)) > 1E-6:# Are the normals to the faces linearly dependent? (N.B. this means that in 3D we'll only diffract from corners, not edges)
                    for d in dirs: # For each direction,
                        for vf, n in zip(vfs, ns):
                            if np.dot(d, n) < 0: # find a face that we can start tracing from
                                fd = problem.mesh.directions[vf]
                                fc = fd[0] + np.sum(fd[1:(dim+1)], axis=0)/dim 
                                p =  vp * (1-offset) + fc * offset #pick a point on the face close to the vertex
                                diffractedpoints.append(TracePoint(vf, p, d)) # todo: probably need to offset the point
                                break
                vertices.add(v)
#    print diffractedpoints
    return diffractedpoints

@put.print_timing        
def tracemesh(problem, sources):
    tracer = pre.HomogenousTrace(problem.mesh, sources.keys())
    tracepoints = []
    for bdy, inidirs in sources.iteritems():
        tracepoints.extend(getstartingtracepoints(problem, bdy, inidirs, 10))
    
    rt = RayTracing(problem.mesh, tracepoints, tracer)
    etods = rt.getDirections()
    diffpoints = processreflections(problem, rt.getReflections())
    rtdiff = RayTracing(problem.mesh, diffpoints, tracer)
#    print rtdiff.getDirections()
    etodcombine(etods, rtdiff.getDirections())
    return [np.array(ds).reshape(-1, problem.mesh.dim) for ds in etods]
    #return dict([(e, np.array(ds).reshape(-1, problem.mesh.dim)) for e, ds in enumerate(etods) if len(ds)])

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
        
     