'''
Created on Nov 25, 2011

@author: joel
'''

import rsf.api as ra
import numpy as np
import pylab
import pypwdg.utils.file as puf

def readvel():
    with(puf.pushd('data')):
        f = ra.Input("marmvel.rsf")
    info = dict([(name, f.float(name)) for name in ['d1', 'd2']]+[(name, f.int(name)) for name in ['n1', 'n2']])
    a = np.empty(f.shape(), order='F', dtype='float32').transpose()
    f.read(a)
    return a.transpose(), info

def showvel():
    pylab.imshow(readvel())
    
print __name__

if __name__ == "__main__":
    showvel()
    pylab.show()