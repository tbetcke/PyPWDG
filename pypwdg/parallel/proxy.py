'''
Created on Sep 14, 2010

@author: joel
'''
import sys

workerobjects = {}
    
def createproxy(klass, id, *args, **kwargs):
    subject = klass(*args, **kwargs)
    workerobjects[id] = subject

def registerproxy(id, subject):
    workerobjects[id] = subject
    

class Proxy(object):
    """ A Proxy object delegates calls to an underlying subject.
        
        Subjects are identified by unique id.  When a proxy is deserialised, it looks up the subject in the 
        workerobjects dictionary.
        
        Nothing clever is done on object deletion.  It probably ought to be ...
    """
    def __init__( self, klass, id, subject = None):           
        self.__klass = klass
        self.__id = id
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
        return (self.__klass, self.__id)
    
    def __setstate__(self, state):
        self.__klass, self.__id = state
        self.subject = workerobjects.get(self.__id)
        
    def __del__(self):
        if self.subject is None:
            print "Proxy has been collected"
            
        