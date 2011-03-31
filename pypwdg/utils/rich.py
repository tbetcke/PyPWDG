'''
Created on 29 Mar 2011

@author: rdodd
'''

import cPickle
import sys

def write_out(array, filename):
    with open(filename, "a") as f:
        cPickle.dump(array, f)
    
