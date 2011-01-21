'''
Created on Jan 19, 2011

@author: joel
'''
import pypwdg.parallel.decorate as ppd

class elementddictinfo(object):
    def __init__(self, mesh):
        self.mesh = mesh        
        
    def getOwnedKeys(self):
        return self.mesh.partition
    
    def getUnownedKeys(self):
        return self.mesh.neighbourelts

def prescatteredargs(n):
    def nullscatterer(data):
        assert len(data) == n
        return data

@ppd.distribute()
class ddict(object):
    
    def __init__(self, ddictinfo):
        self.ownedkeys = set(ddictinfo.getOwnedKeys())
        self.unownedkeys = ddictinfo.getUnownedKeys()
        self.changeddata = {}
        self.owneddata = {}
        self.unowneddata = {}
    
    def __setitem__(self, key, value):
        assert key in self.ownedkeys
        self.owneddata[key] = value
        self.changeddata[key] = value
        
    def __getitem__(self, key):
        return self.owneddata[key] if key in self.ownedkeys else self.unowneddata[key]
    
    @ppd.parallelmethod(None, None)
    def getChangedData(self):
        changeddata = self.changeddata.copy()
        self.changeddata.clear()
        return changeddata
    
    @ppd.parallelmethod(None, None)
    def getUnownedKeys(self):
        return self.unownedkeys 
    
    @ppd.parallelmethod(prescatteredargs, None)
    def setUnownedData(self, data):
        self.unownneddata.update(data)
    
class ddictmanager(object):
    
    def __init__(self, ddictinfo):
        self.ddict = ddict(ddictinfo)
        self.unownedkeys = self.ddict.getUnownedKeys()
        self.datacopy = {}
    
    def getDict(self):
        return self.ddict
    
    def sync(self):
        newdata = self.ddict.getChangedData()
        for data in newdata: self.datacopy.update(data)
        unowneddata = [dict(zip(keys, [newdata[key] for key in keys])) for keys in self.unownedkeys]
        self.ddict.setUnownedData(unowneddata)
        
        
        