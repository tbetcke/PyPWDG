'''
Created on Nov 25, 2011

@author: joel
'''

import rsf.api as ra
import numpy as np
import pylab

def readvel():
    file = ra.Input("marmvel.rsf")
    a = np.empty(file.shape(), order='F', dtype='float32').transpose()
    file.read(a)
    return a.transpose()

def showvel():
    pylab.imshow(readvel())
    
print __name__

if __name__ == "__main__":
    showvel()
    pylab.show()