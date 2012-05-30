'''
Created on May 30, 2012

@author: joel
'''
import pypwdg.parallel.decorate as ppd
import pypwdg.mesh.structure as pms

@ppd.distribute()
class NormEvaluator(object):
    def __init__(self, volumeassembly, mesh):
        E = pms.ElementMatrices(mesh)
        self.L2 = volumeassembly.assemble([[E.I, E.Z],[E.Z, E.Z]]).tocsr()
    
    @ppd.parallelmethod()
    def l2norm2(self, x):
        l2 = x * self.L2 * x
        return l2
        
        
@ppd.distribute()
class CINormEvaluator(NormEvaluator):
    def __init__(self, computationinfo):
        volumeassembly = computationinfo.volumeAssembly()
        NormEvaluator.__init__(self, volumeassembly, computationinfo.problem.mesh)


@ppd.distribute()
class SPNormEvaluator(NormEvaluator):
    def __init__(self, mesh, structuredpoints):
        pass

@ppd.distribute()
class ErrorEvaluator(object):
    def __init__(self, norm2, truesoln, operator = None):
        self.norm2 = norm2
        self.truesoln = truesoln
        self.lift = operator.postprocess if operator is not None and hasattr(operator, 'postprocess') else lambda x:x
    
    @ppd.parallelmethod()
    def evaluate(self, x):
        d = x - self.truesoln
        ld = self.lift(d)
        e = self.norm2(ld)
        return e
        
            
