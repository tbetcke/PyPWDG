'''
Created on Mar 13, 2012

@author: joel
'''
import pypwdg.mesh.submesh as pmsm
import pypwdg.parallel.decorate as ppd
import pypwdg.setup.computation as psc
import pypwdg.core.boundary_data as pcbd
import copy    
            
''' Domain decomposition iterative procedure:
    - Create sub problems.  
    - Subproblems get boundary data from a mortar variable
'''


class DDWorker(object):
    def __init__(self, system, etomortarbasis, sysargs, syskwargs):
        self.system = system
        self.S,self.G = system.getSystem(*sysargs, **syskwargs)
        self.B = system.getBoundary('INTERNAL', etomortarbasis)
        
    def fn(self):
        
        bdyinfo = (self.bcoeffs, bdyetob)
        Sinternal = system.getBoundary('INTERNAL', bdyinfo)
    
class DDSolver(object):
    def solve(self, system, bcoeffs, sysargs, syskwargs):
        
    

class DDComputation(psc.Computation):
    def __init__(self, problem, basisrule, systemklass, *args, **kwargs):
        submesh = pmsm.SubMesh(problem.mesh, 'INTERNAL')
        localproblem = copy.copy(problem)
        localproblem.mesh = submesh
        localproblem.bdyinfo["INTERNAL"] = pcbd.zero_impedance(problem.k):
        psc.Computation.__init__(self, localproblem, basisrule, systemklass, *args, **kwargs)
         
        
        