'''
Created on Apr 12, 2011

@author: joel
'''
import pypwdg.core.bases.utilities as pcbu
import pypwdg.core.bases.variable as pcbv
import pypwdg.parallel.decorate as ppd
import pypwdg.parallel.distributeddict as ppdd


class Problem(object):
    ''' This class defines a (Helmholtz) problem to be solved'''
    def __init__(self, mesh, k, bndconds):
        self.mesh = mesh
        self.k = k
        self.bdyinfo = {entity: (bdycond.coeffs, pcbu.UniformFaceToBases(bdycond.data, mesh)) for entity, bdycond in  bndconds.items()}
    
    def populateBasis(self, etob, basisrule):
        ''' Helper function to initialise the element to basis map in each partition'''  
        ei = self.elementinfo
        for e in self.mesh.partition:
            etob[e] = basisrule.populate(ei.info(e))
    
    elementinfo = property(lambda self:pcbu.ElementInfo(self.mesh, self.k))


class VariableNProblem(Problem):
    def __init__(self, entityton, mesh, k, bnddata):
        Problem.__init__(self, mesh, k, bnddata)
        self.entityton = entityton
        print "Warning: variable n not incorporated into boundary info " #todo: fix this.  
        
    elementinfo = property(lambda self: pcbv.EntityNElementInfo(self.mesh,self.k,self.entityton))
    
@ppd.parallel(None, None)
def localPopulateBasis(etob, basisrule, problem):
    problem.populateBasis(etob, basisrule)

def constructBasis(problem, basisrule):
    ''' Build an element to basis (distributed) map based on a basisrule'''
    manager = ppdd.ddictmanager(ppdd.elementddictinfo(problem.mesh), True)
    etob = manager.getDict()
    localPopulateBasis(etob, basisrule, problem)
    manager.sync()   
    return pcbu.ElementToBases(etob, problem.mesh)

