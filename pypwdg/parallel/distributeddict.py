'''
A framework for maintaining a distributed dictionary using MPI.

Distributed dictionaries have an essentially immutable set of keys (the immutability isn't
enforced, but it's recommended).  Within each process, some of these keys are "owned", 
meaning that that process is responsible for setting the values associated with the key.  
Each process also has access to the values for "unowned" keys (e.g. the data on the boundary
of a neighbouring process).  

ddicts are controlled by a ddictmanager.  Calling .sync() on the ddictmanager will ensure that
all values associated with unowned keys in each process are updated.  The ddictmanager 
optionally maintains a master copy of all the data.

Each ddict needs to be told what its owned and unowned keys are.  This is done using by a
dictinfo object, which must have getOwnedKeys() and getUnownedKeys() methods, which return 
the correct information for the current process.      

For the future: 
- Some kind of automatic sync-ing is possible, hooking directly into the messaging
infrastructure.  It would, however, add some overhead.  Not sure whether it's worth it.
- Automatic determination of unowned keys is also possible, although to be efficient,
that requires inter-worker-process communication, which we currently avoid.  It could
be done using MPI Windows, however Windows are not designed for unstructured data.    

Created on Jan 19, 2011

@author: joel
'''
import pypwdg.parallel.decorate as ppd
from pypwdg.parallel.mpiload import mpiloaded

class elementddictinfo(object):
    """ Provides the info for a distributed dictionary with mesh elements as keys.
    
        Owned keys are those in the partition for this process.  Unownedkeys are 
        neighbouring elements.
    """
    
    def __init__(self, mesh):
        self.mesh = mesh        
        
    def getOwnedKeys(self):
        return self.mesh.partition
    
    def getUnownedKeys(self):
        return self.mesh.neighbourelts

def prescatteredargs(n):
    def nullscatterer(obj, data):
        assert len(data) == n
        return [((obj, d), {}) for d in data]
        return data
    return nullscatterer

def combinedict(d1, d2):
    d1.update(d2)
    return d1

@ppd.distribute()
class ddict(dict):
    """ A distributed dictionary.  
    
        Keeps track of the values associated with owned keys
    """
    def __init__(self, ddictinfo):
        self.ownedkeys = ddictinfo.getOwnedKeys()
        self.unownedkeys = ddictinfo.getUnownedKeys()
        self.lastsync = [None]*len(self.ownedkeys)
            
    @ppd.parallelmethod(None, combinedict)
    def getChangedData(self):
        changeddata = dict([(k,self.get(k)) for k,lv in zip(self.ownedkeys, self.lastsync) if self.get(k)!=lv] )
        self.lastsync = [self.get(k) for k in self.ownedkeys]
        return changeddata
    
    @ppd.parallelmethod(None, None)
    def getUnownedKeys(self):
        return self.unownedkeys 
    
    @ppd.parallelmethod(prescatteredargs, None)
    def setUnownedData(self, data):
        self.update(data)
    
class ddictmanager(object):
    """ Create and manage a ddict.
    
        ddictinfo: An object that, when passed to the worker processes, will tell them
        what entries they own and what entries they are interested in
        
        localcopy: Optional dict that maintains a local copy (on the master process) of
        all the data.
    """  
    
    def __init__(self, ddictinfo, localcopy = None):
        self.ddict = ddict(ddictinfo)
        self.unownedkeys = self.ddict.getUnownedKeys()
        self.datacopy = localcopy                    

    def getDict(self):
        """ Return the managed ddict"""
        return self.ddict
        
    def sync(self):
        """ Sync the managed ddict across all processes"""
        newdata = self.ddict.getChangedData()
        if self.datacopy is not None: self.datacopy.update(newdata)
        unowneddata = [dict(zip(keys, [newdata[key] for key in keys])) for keys in self.unownedkeys]
        self.ddict.setUnownedData(unowneddata)
        
        
        