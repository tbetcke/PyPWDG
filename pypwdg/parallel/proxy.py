'''
Created on Sep 14, 2010

@author: joel
'''

workerobjects = {}

def createproxy(klass, id, *args, **kwargs):
    workerobjects[id] = Proxy(klass, id, klass(args, kwargs))

class Proxy(object):
    def __init__( self, klass, id, subject = None):                
        self.__klass = klass
        self.__id = id
        self.__subject = subject
        
    def __getattr__( self, name ):
        print "__getattr__(%s)"%name
        if self.__subject is None:        
            attr =  getattr( self.__klass, name )
            return lambda *arg, **kwargs: attr(self, *arg, **kwargs)
        else:
            return getattr(self.__subject, name)
    
    def __getstate__(self):
        return (self.__klass, self.__id)
    
    def __setstate__(self, state):
        self.__klass, self.__id = state
        self.__subject = locals.get(self.__id)
