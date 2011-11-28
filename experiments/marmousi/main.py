'''
Created on Nov 26, 2011

@author: joel
'''

import pypwdg.test.utils.mesh as ptum
import scipy.interpolate as si
import utils
import numpy as np
import pypwdg.core.bases as pcb
import pypwdg.core.boundary_data as pcbd
import pypwdg.setup.computation as psc
import pypwdg.setup.problem as psp
import pypwdg.core.physics as pcp
import pypwdg.core.bases.variable as pcbv
import pypwdg.output.solution as pos
import matplotlib.pyplot as mp
import pypwdg.output.mploutput as pom

data, info = utils.readvel()
dy = info['d1']
dx = info['d2']
ny = info['n1']
nx = info['n2']
data = data.T
print nx,ny,dx,dy
print data.shape
bounds = [[0,dx*nx], [0,dy*ny]]  
H = 50          
h = H * dx

k = 1
npw = 3
quadpoints = 4
g = pcb.FourierHankel([nx*dx/2,-ny*dy/10], [0], k)
impbd = pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g)

averagev = np.average(data)
averagek = averagev * k
#velocity = si.interp2d(np.arange(n1)*d1, np.arange(n2)*d2, data.flatten(), copy=False, bounds_error=False, fill_value=averagev)
s = si.RectBivariateSpline(np.arange(nx)*dx, np.arange(ny)*dy, data, kx=1,ky=1)
velocity = lambda p: s.ev(p[:,0],p[:,1])
pom.output2dfn(bounds, velocity, [nx,ny]) # yes, I know that it's upside-down
entityton = {4:velocity}
mesh = ptum.regularrectmesh(bounds[0], bounds[1], nx/H, ny/H)
print mesh.nelements
bnddata = dict(enumerate([impbd]*4)) 

problem=psp.VariableNProblem(entityton, mesh,k, bnddata)

pdeg = 1
alpha = ((pdeg*1.0)**2 / h)  / averagek
beta = averagek / (pdeg * 1.0*h) 

pw = pcbv.PlaneWaveVariableN(pcb.circleDirections(npw))
basisrule = pw
computation = psc.Computation(problem, basisrule, pcp.HelmholtzSystem, quadpoints, alpha = alpha, beta = beta)
solution = computation.solution(psc.DirectSolver().solve, dovolumes=True)
pos.standardoutput(computation, solution, quadpoints, bounds, np.array([nx,ny]), 'marmousi', mploutput=True)
mp.show()


