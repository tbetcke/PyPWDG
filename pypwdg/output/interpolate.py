'''
Created on Feb 1, 2012

@author: joel
'''
import numpy as np
import pypwdg.utils.geometry as pug
import scipy.interpolate as si
import datetime

class InterpolatedSolution(object):
    ''' Captures the solution data at interpolation points, so it can be evaluated as a function
        Importantly, this class is picklable
    '''
    def __init__(self, solution, bounds, npoints, customdescription=''):
        bounds=np.array(bounds,dtype='d')
        self.sp = pug.StructuredPoints(bounds.transpose(), npoints)
        spe = solution.getEvaluator()        
        vals, counts = spe.evaluate(self.sp)
        counts[counts==0] = 1
        self.values = vals / counts
        
        
        self.description="%s k=%s %s"%(customdescription,solution.problem.k, datetime.datetime.today())
    
    def __call__(self, points):
        return si.griddata(self.sp.toArray(), self.values, points)
        
    
        