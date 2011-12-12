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
import pypwdg.core.bases.reference as pcbr

import pypwdg.parallel.main

npw = 15
quadpoints = 4
pdeg = 1
usepoly = False

effectivek = 30
effectiveh = 0.05
veldata = utils.RSFVelocityData(exteriorvel = None)
[[xmin,xmax],[ymin,ymax]] = veldata.bounds
#effectivek = omega * distance / vel.averagevel
omega = effectivek * veldata.averagevel / (ymax - ymin)
averagek = effectivek / (ymax - ymin)
trueh = effectiveh / (ymax - ymin)
print 'omega = %s'%omega
print 'bounds %s'%veldata.bounds
print veldata.averagevel
npoints =  np.array([veldata.nx,veldata.ny])
#pom.output2dfn(veldata.bounds, veldata, npoints)


sourcexpos = (xmax + xmin) / 2
sourcek = veldata(np.array([[sourcexpos,0]]))[0] * averagek
g = pcb.FourierHankel([sourcexpos,0], [0], sourcek)
sourceimp = pcbd.generic_boundary_data([-1j*averagek,1],[-1j*averagek,1],g)
zeroimp = pcbd.zero_impedance(averagek)

entityton = {4:veldata}
mesh = ptum.regularrectmesh([xmin,xmax],[ymin,ymax], int(((xmax - xmin) / (ymax - ymin)) / effectiveh), int(1 / effectiveh))
print mesh.nelements
bnddata = {0:sourceimp,1:zeroimp,2:zeroimp,3:zeroimp} 
#bdndata = {0:impbd, 1:pcbd.zero_impedance}

problem=psp.VariableNProblem(entityton, mesh,averagek, bnddata)

alpha = ((pdeg*1.0)**2 / trueh)  / averagek
beta = averagek / (pdeg * 1.0* trueh) 

pw = pcbv.PlaneWaveVariableN(pcb.circleDirections(npw))
if usepoly:
    poly = pcbr.ReferenceBasisRule(pcbr.Dubiner(pdeg))
    polypw = pcb.ProductBasisRule(poly, pw)
    basisrule = polypw
else:
    basisrule = pw

computation = psc.Computation(problem, basisrule, pcp.HelmholtzSystem, quadpoints, alpha = alpha, beta = beta, usecache=False)
solution = computation.solution(psc.DirectSolver().solve, dovolumes=True)
pos.standardoutput(computation, solution, quadpoints, veldata.bounds, npoints, 'marmousi%s-%s-%s-%s'%(effectivek,effectiveh, npw, pdeg if usepoly else 0), mploutput=True)
#pom.output2dsoln(veldata.bounds, solution, np.array([veldata.nx,veldata.ny]), plotmesh = False)
mp.show()


