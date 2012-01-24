'''
Created on Nov 25, 2011

@author: joel
'''

import rsf.api as ra
import numpy as np
import pypwdg.output.mploutput as pom
import pypwdg.utils.file as puf
import scipy.interpolate as si


class RSFVelocityData():
    def __init__(self, exteriorvel='average'):
        ''' Get the (marmousi) velocity data.  
        
        Arguments:
            exteriorvel: add a layer around the edge with a particular velocity.  'average' takes the average
        '''
        data, info = readvel()
        data = data.T
        self.dy = info['d1']
        self.dx = info['d2']
        self.ny = info['n1']
        self.nx = info['n2']
        self.bounds = [[0.0,self.dx*self.nx], [0.0,self.dy*self.ny]]
        self.averagevel = np.average(data)
        if exteriorvel:
            if exteriorvel == 'average': exteriorvel = self.averagevel
            dd = np.ones(np.array(data.shape) + [2,2])
            dd[1:-1,1:-1] = data
            x0 = -1
            xend = self.nx + 1
            y0 = -1
            yend = self.ny + 1    
        else:
            dd = data
            x0 = 0
            xend = self.nx
            y0 = 0
            yend = self.ny
        self.s = si.RectBivariateSpline(np.arange(x0,xend)*self.dx, np.arange(y0,yend)*self.dy, dd, kx=1,ky=1)
    
    
    def __call__(self, p):
        return self.averagevel / self.s.ev(p[:,0],p[:,1])

    def show(self):
        pom.output2dfn(self.bounds, self, [self.nx,self.ny])
    


def readvel():
    with(puf.pushd('data')):
        f = ra.Input("marmvel.rsf")
    info = dict([(name, f.float(name)) for name in ['d1', 'd2']]+[(name, f.int(name)) for name in ['n1', 'n2']])
    a = np.empty(f.shape(), order='F', dtype='float32').transpose()
    f.read(a)
    return a.transpose(), info
