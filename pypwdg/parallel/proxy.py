'''
Proxy objects are designed to allow method calls to be distributed across MPI processes (via
ppd.distribute or ppd.immutable) 

If the proxy is collected on the master process, it gets collected on the worker processes.  This
shouldn't be a big deal.  If the master object gets collected, there's no way for the worker objects
to communicated.  You shouldn't have used a Proxy! (N.B. this is a little unfair - it's possible there might
be different usage patterns across different applications.  In any case - keep the master proxy alive and all 
will be well.

There's an important, but hopefully benign limitation: Returning a proxy back to the master process will
possible cause odd things to happen.  Since it's hard to see why you would do this (how do you reduce
the function return value?) hopefully it's not a big deal either.

Created on Sep 14, 2010

@author: joel
'''
from pypwdg.parallel.mpiload import mpiloaded, comm
import pypwdg.parallel.messaging as ppm
import weakref
import uuid
import logging

workerobjects = {}
uidweakrefs = weakref.WeakKeyDictionary()

def getnewproxyuid():
    ''' Get a new proxy id.  Register a callback against it, so that when it gets collected, we can clean
    up the worker processes'''
    uid = uuid.uuid4()
    uidint = uid.int
    def uidcallback(uidref):
        if mpiloaded and comm.rank == 0:
            logging.log(logging.INFO, 'Distributing unregister for %s'%(uidint))
            ppm.asyncfncall(unregisterproxy, [((uuid.UUID(int = uidint),),{})] * (comm.size-1))        
    uidweakrefs[uid] = weakref.ref(uid, uidcallback)
    return uid
    

def createproxy(klass, uid, *args, **kwargs):
    subject = klass(*args, **kwargs)
    workerobjects[uid] = subject

def registerproxy(uid, subject):
    workerobjects[uid] = subject

def unregisterproxy(uid):
    logging.log(logging.INFO, 'Unregistering %s'%(uid))
    logging.log(logging.DEBUG, 'Remaining worker objects: %s'%(len(workerobjects)))
    del workerobjects[uid]
      

class Proxy(object):
    """ A Proxy object delegates calls to an underlying subject.
        
        Subjects are identified by unique id.  When a proxy is deserialised, it looks up the subject in the 
        workerobjects dictionary.
        
        Nothing clever is done on object deletion.  It probably ought to be ...
    """
    def __init__( self, klass, subject = None):
        self.__klass = klass
        self.__id__ = getnewproxyuid()
        self.subject = subject
        
    def __getattr__( self, name ):
        if self.subject is None:        
            raise AttributeError("This proxy has no subject and no distributed attribute called %s"%name)
        return self.subject.__getattribute__(name)
    
    def __setitem__(self, key, value):
        if self.subject is None:
            raise TypeError("This proxy has no subject and so does not support item assignment")
        self.subject[key] = value
        
    def __getitem__(self, key):
        if self.subject is None:
            raise TypeError("This proxy has no subject and so is not subscriptable")
        return self.subject[key]
    
    def __getstate__(self):
        return (self.__klass, self.__id__)
    
    def __setstate__(self, state):
        self.__klass, self.__id__ = state
        self.subject = workerobjects.get(self.__id__)
            
