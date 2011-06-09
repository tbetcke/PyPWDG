'''
Created on Jun 8, 2011

@author: joel
'''
import pypwdg.core.bases.definitions as pcbd
import numpy as np



class SVDBasisReduceRule(object):
    ''' Uses the SVD to reduce a basis using a given L^2 norm
    
        refquad: the quadrature to base the L^2 norm on (on the reference element)
        rule: rule to give the underlying basis
        threshold: cut-off point below which singular values are discarded (N.B. the basis is normalised first)
    '''         
    def __init__(self, refquad, rule, threshold = 1E-5):
        self.rule = rule
        self.refp, self.refw = refquad
        self.threshold = threshold
    
    def populate(self, einfo):
        basis = pcbd.BasisCombine(self.rule.populate(einfo)) # get the underlying basis
        points = einfo.refmap.apply(self.refp) # work out the quadrature points on this element
        weights = np.abs(einfo.refmap.det(self.refp)) * self.refw # and the quadrature weights
        values = basis.values(points) * np.sqrt(weights.reshape(-1,1)) # evaluate the underlying basis ...
        normalisedvalues = values / np.sqrt(np.sum(values*values.conjugate(), axis=0)) # ... and normalise it
        u,s,vh = np.linalg.svd(normalisedvalues, full_matrices=False) # calculate the SVD
        n = (s < self.threshold).argmax() # find the index of the first singular value below the threshold
        if n == 0 : 
            n = len(s)
        else :
            print "Reducing basis from ",len(s)," to ",n
        M = s.reshape(-1,1)[:n] * vh.conjugate()[:n] # this is a tad confusing.  BasisReduce accepts the transpose of the post-multiplication matrix, so we only need to do the conjugate here.
        return [pcbd.BasisReduce(basis, M)]
        
        
        