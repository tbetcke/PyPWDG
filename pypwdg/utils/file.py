'''
Created on Nov 26, 2011

@author: joel
'''
import os
from contextlib import contextmanager

@contextmanager 
def pushd(directory): 
    ''' temporarily change the current directory.
    
        usage:
        
        with(pushd('some/other/directory')):
            # do some file access
        
        # now we're back in the original working directory
    
        inspired by http://software.clapper.org/grizzled-python/epydoc/grizzled.os-pysrc.html
    '''
    
    cwd = os.getcwd() 
    try: 
        os.chdir(directory) 
        yield directory 
    finally: 
        os.chdir(cwd) 