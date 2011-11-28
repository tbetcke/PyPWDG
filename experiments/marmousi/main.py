'''
Created on Nov 26, 2011

@author: joel
'''

import pypwdg.test.utils.mesh as ptum
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

effectivek = 80
effectiveh = 0.1
veldata = utils.RSFVelocityData()
[[xmin,xmax],[ymin,ymax]] = veldata.bounds
#effectivek = omega * distance / vel.averagevel
omega = effectivek * veldata.averagevel / (ymax - ymin)
averagek = omega / veldata.averagevel
print 'omega = %s'%omega

npw = 3
quadpoints = 4
g = pcb.FourierHankel([(xmax + xmin) / 2,ymin - (ymin - ymax)/10], [0], averagek)
impbd = pcbd.generic_boundary_data([-1j*averagek,1],[-1j*averagek,1],g)

entityton = {4:veldata}
mesh = ptum.regularrectmesh([xmin,xmax],[ymin,ymax], veldata.nx * veldata.dx / effectiveh, veldata.ny * veldata.dy / effectiveh)
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


