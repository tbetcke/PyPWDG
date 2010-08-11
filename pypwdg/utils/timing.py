'''
Created on Aug 6, 2010

@author: joel
'''
import time

def print_timing(func):
    """ timing utility function.  Use @print_timing """
    def wrapper(*arg, **kwds):
        t1 = time.time()
        res = func(*arg, **kwds)
        t2 = time.time()
        print '%s took %0.3f ms' % (func.func_name, (t2-t1)*1000.0)
        return res
    return wrapper

class Timer(object):
    
    def start(self):
        self.ts = [time.time()]    
        self.desc = []
        return self
    
    def split(self, msg = None):
        self.desc.append("Split %s"%len(self.ts)  if msg == None else msg)
        self.ts.append(time.time())
        
    def show(self):
        for t0,t1,desc in zip(self.ts[:-1], self.ts[1:], self.desc):
            print desc + ": %s ms"% ((t1 - t0) * 1000)
        