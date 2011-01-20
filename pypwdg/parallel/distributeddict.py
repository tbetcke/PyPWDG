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

@ppd.distribute()
class ddict(object):
    
    def __init__(self, ddictinfo):
        self.ddictinfo = ddictinfo
        self.data = {}
    
    @ppd.parallelmethod(None, None)
    def getOwnedData(self):
        return self.dict
    
    @ppd.parallelmethod(None, None)
    def getUnownedKeys(self):
        return self.ddictinfo.getUnownedKeys() 
    
class ddictmanager(object):
    
    def __init__(self, ddictinfo):
        self.ddict = ddict(ddictinfo)
        self.unownedkeys = self.ddict.getUnownedKeys()
        self.datacopy = {}
    
    def getDict(self):
        return self.ddict
    
    def sync(self):
        newdata = self.ddict.getOwnedData()