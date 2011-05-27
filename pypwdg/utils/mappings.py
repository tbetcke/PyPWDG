'''
Affine maps.  Stolen from pypyramid.  

@author: joel
'''
import numpy as np
import numpy.linalg as nl

class Affine(object):
    """ An affine map """
    def __init__(self, offset, linear, inverse = None):
        self.offset = offset.reshape(1,-1)
        self.linear = linear
        self.__inverse = inverse      
    
    def apply(self, p):
        return (np.dot(p, self.linear) + self.offset)
        
    def __calcinverse(self):
        if self.__inverse == None:
            invlinear = nl.inv(self.linear)
            invoffset = -np.dot(self.offset, invlinear)
            self.__inverse = Affine(invoffset, invlinear, self)
        return self.__inverse
    
    inverse = property(__calcinverse)

#    
#def buildaffine(pfrom, pto):
#    # treat the first point as the origin in each case
#    # Taking transposes means that the first axis is the x,y,z component of the points in the second axis.
#    if len(pfrom) > 1:
#        F = (pfrom[1:] - pfrom[0]).transpose()
#        T = (pto[1:] - pto[0]).transpose()
#        
#        # we want to find M such that M . F = T
#        # if not enough points have been supplied, add in some orthogonal ones
#        fmissing = F.shape[0] - F.shape[1]
#        if fmissing:
#            F = np.hstack((F, np.zeros((F.shape[0], fmissing))))
#            T = np.hstack((T, np.zeros((T.shape[0], fmissing))))
#            FQ = nl.qr(F)[0]
#            TQ = nl.qr(T)[0]
#            F[:,-fmissing:] = FQ[:,-fmissing:]
#            T[:,-fmissing:] = TQ[:,-fmissing:]
#        
#        M = nl.solve(F.transpose(),T.transpose())
#        offset = pto[0] - np.dot(pfrom[0], M)
#        return Affine(offset, M.transpose())
#    