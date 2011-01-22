'''
Created on Nov 13, 2010

@author: joel
'''
import pypwdg.parallel.decorate

from pypwdg.parallel.mpiload import *

import numpy as np
import cPickle
import cStringIO
import time

import operator

comm = mpi.COMM_WORLD

class ArrayHandler(object):
    """ An ArrayHandler is used in conjunction with the pickle persistency mechanism to collect numpy arrays so that they
        may be sent over MPI using the native serialisation.
        
        uid: Arrays are replaced with a persistent id.  uid is the prefix.
        dtype: Type of array to look for
        minlen: It's probably not worth doing this for short arrays (although I haven't verified this).  Anything shorter than minlen is ignored
        nexthandler: ArrayHandlers may be chained together (e.g. so that arrays of more than one datatype can be extracted)
        
        The ArrayHandler handles both ends of the communication.  Strategy for use:
            
            // On sending process (N):
            myhandler = ArrayHandler("myuid", dtype, minlen, someotherhandler)    
            
            buf = ...
            p = cPickle.Pickler(buf, ...)
            p.persistent_id = myhandler.process
            p.dump(obj) // obj is the object tree that we're sending                                        
            comm.send((buf.getvalue(), myhandler), dest=M)  // or whatever you want to do to send the main message.  Note that only 
                                                            // one handler needs to be sent explicitly.  Any chained handlers will 
                                                            // automatically be serialised.
            myhandler.send(N) // sends the arrays that we extracted from obj
            
            // On receiving process:
            bufvalue, myhandler = comm.receive(N)
            myhandler.receive() // receives the array data, reconstructs all the arrays (Note that the handler knows which process it was send from)
            buf = ... create a buffer out of bufvalue
            up = cPickle.Unpickler(buf)
            up.persistent_load = myhandler.lookup
            obj = up.load()
             
    """
    def __init__(self, uid, dtype, minlen, nexthandler = None):
        self.id = 0
        self.uid = uid
        self.dtype = dtype
        self.minlen = minlen
        self.source = comm.rank
        self.nexthandler = nexthandler
        self.objtosid = {}
        self.sidtoobj = {}
        self.sids = []
        self.shapes = []
    
    def process(self, obj):    
        """ This is the persistent_id method
        
            If obj matches the criteria for this ArrayHandler, a unique id will be returned instead.  
            arrays are remembered, but this is just done at an object level so a and a.ravel() will be serialised
            as two separate objects.  
            
            If obj does not match and there's a nexthandler, returns nexthandler.process(obj), otherwise returns None
        """
        if isinstance(obj, np.ndarray):
            if obj.dtype == self.dtype and obj.size >= self.minlen:
                sid = self.objtosid.get(id(obj))
                if sid is None:
                    sid = self.uid+str(self.id)
                    self.sidtoobj[sid] = obj
                    self.objtosid[id(obj)]= sid
                    self.sids.append(sid)
                    self.shapes.append(obj.shape)
                    self.id+=1
                return sid        
        return None if self.nexthandler is None else self.nexthandler.process(obj) 
    
    def lookup(self, sid):
        """ This is the persistent_loopup method.  If sid is recognised, the appropriate array is returned"""
        
        obj = self.sidtoobj.get(sid)
        if obj is None:
            if self.nexthandler is not None:
                obj = self.nexthandler.lookup(sid)
        return obj
    
    def send(self, dest):
        """ Send the array data to dest.  Also tells nexthandler to send"""
        if len(self.sids):
            a = np.concatenate(map(np.ravel,[self.sidtoobj[sid] for sid in self.sids]))
            print "sending %s array of length %s"%(self.dtype, len(a))
            comm.Send(a,dest)
        if self.nexthandler is not None: self.nexthandler.send(dest)
    
    def receive(self):
        """ Receive the array data from self.source."""
        if len(self.sids):
            sizes = [reduce(operator.mul, shape) for shape in self.shapes]
            totallen = sum(sizes)
            print "receiving %s array of length %s"%(self.dtype, totallen)
            a = np.empty(totallen, dtype = self.dtype)
            comm.Recv(a, self.source)
            ixs = np.cumsum([0]+sizes)     
            self.sidtoobj = dict([(sid, a[i0:i1].reshape(shape)) for sid, i0,i1,shape in zip(self.sids, ixs[:-1],ixs[1:],self.shapes)])
        else:
            self.sidtoobj = {}
        if self.nexthandler is not None: self.nexthandler.receive()
         
    
    def __getstate__(self):
        return (self.id, self.dtype, self.minlen, self.source, self.sids, self.shapes, self.nexthandler)
    
    def __setstate__(self, state):
        self.id, self.dtype, self.minlen, self.source, self.sids, self.shapes, self.nexthandler = state  

def mastersend(objs):
    for dest in range(1,comm.size):
        comm.send(None, dest)
    comm.scatter(objs, root=0)

def workerrecv():
    buf = np.array([0])
    req = comm.Irecv(buf)
    while(True):
        s = mpi.Status()
        req.Get_status(s)
        if s.Get_source()==0: break
        time.sleep(0.001)
    obj = comm.scatter(root=0)
    return obj

def workersend(obj):
    realhandler = ArrayHandler("real", np.float, 50, None)
    complexhandler = ArrayHandler("cplx", np.complex, 25, realhandler)    
    csio = cStringIO.StringIO()
    p = cPickle.Pickler(csio, protocol=2)
    p.persistent_id = complexhandler.process
    p.dump(obj)            
    comm.gather((csio.getvalue(), complexhandler), None, root=0)
    complexhandler.send(0)

def masterrecv():
    gv = comm.gather(root = 0)[1:]
    values = []
    for s, handler in gv:
        handler.receive()
        up = cPickle.Unpickler(cStringIO.StringIO(s))
        up.persistent_load = handler.lookup
        values.append(up.load())
    return values

def scatterfncall(fn, args, reduceop=None):
    """ Scatter function calls to the worker processes
    
        fn: function to be scattered (must be picklable)
        args: list of (*args, *kwargs) pairs - same length as number of workers
        reduceop: reduction operator to apply to results (can be None)        
    """

#    mpi.broadcast(mpi.world, root=0)
    
    tasks = [None] # task 0 goes to this process, which we want to remain idle.       
    # generate the arguments for the scattered functions         
    for a in args:
        tasks.append((fn, a[0], a[1]))
    mastersend(tasks)
    values = masterrecv()
    return values if reduceop is None else reduce(reduceop, values)


def workerloop():
        # For some unclear reason, the developers of openmpi think that it's acceptable for a thread to use 100% CPU
        # while waiting at a barrier.  I guess that for a symmetric algorithm with very small work packets, that might
        # be true, but our algorithm is not symmetric (master slave) and the work packets are not small (because
        # we don't expect miracles from mult-processing).  So it's vastly more efficient for our threads to poll
        # every 1ms to see if there's any work to do.    
#            mpi.world.irecv(source=0)
#            while(request.test() is None):
#                time.sleep(0.001)            
    while True:
        task = workerrecv()
        fn, args, kwargs = task
        result = fn(*args, **kwargs) 
        workersend(result)
