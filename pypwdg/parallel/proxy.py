'''
Created on Sep 14, 2010

@author: joel
'''
import pypwdg.parallel.wrapper as ppw

import sys

workerobjects = {}

def createproxybyname(module, name, id, *args, **kwargs):
    __import__(module)
    klass = getattr(sys.modules[module],name)
    return createproxy(klass, id, *args, **kwargs)
    
def createproxy(klass, id, *args, **kwargs):
    subject = klass(*args, **kwargs)
    workerobjects[id] = subject

class Proxy(object):
    def __init__( self, klass, id, subject = None):           
        self.__klass = klass
        self.__id = id
        self.__subject = subject
        
    def __getattr__( self, name ):
        print "__getattr__(%s)"%name
        print self.__klass
        if self.__subject is None:        
            return ppw.methodwrapper(self.__klass,name, self)
        else:
            return self.__subject.__getattribute__(name)
    
    def __getstate__(self):
        return (self.__klass, self.__id)
        #return (self.__klass.__module__, self.__klass.__name__, self.__id)
    
    def __setstate__(self, state):
        self.__klass, self.__id = state
#        module, name, self.__id = state
#        self.__klass = getattr(sys.modules[module],name)
        self.__subject = workerobjects.get(self.__id)
