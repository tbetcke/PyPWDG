'''
Created on Mar 13, 2012

@author: joel
'''
import pypwdg.mesh.submesh as pmsm
import pypwdg.parallel.decorate as ppd

class BoundarySpace(object):
    

class DomainComputation(object):
    def __init__(self, problem, basisrule):
        submesh = pmsm.SubMesh(problem.mesh)
        