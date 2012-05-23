'''
Created on Aug 11, 2010

@author: joel
'''
import pypwdg.mesh.structure as pms
import pypwdg.parallel.decorate as ppd

@ppd.distribute()
class HelmholtzBoundary(object):
    def __init__(self, computationinfo, entity, bdyinfo, delta=0.5):
        bdycoeffs, bdyftob = bdyinfo
        bdyvandermondes = computationinfo.faceVandermondes(bdyftob)  
        rc0,rc1 = bdycoeffs.r_coeffs
        self.loadassembly = computationinfo.faceAssembly(bdyvandermondes, [[rc0,rc1],[rc0,rc1]])
        lc0,lc1 = bdycoeffs.l_coeffs
        self.bdyassembly = computationinfo.faceAssembly(scale=[[lc0,lc1],[lc0,lc1]])
        self.mesh = computationinfo.problem.mesh
        self.B = self.mesh.entityfaces[entity]
        self.delta = delta
        self.entity = entity
        self.k = computationinfo.problem.k
    
    @ppd.parallelmethod()    
    def load(self, collapseload = True):
        ''' The load vector (due to the boundary conditions)'''            
        delta = self.delta
        dik = delta / (1j * self.k)
        B = self.B
        # todo - check the cross terms.  Works okay with delta = 1/2.  
#        GB = self.loadassembly.assemble([[(1-delta) * B, (1-delta) *B], 
#                                    [-dik* B,     -dik *B]])
        GB = self.loadassembly.assemble([[-(1-delta) * B, (1-delta) *B], 
                                    [dik* B,     -dik *B]])
        

        print 'GB', self.entity, GB.tocsr()
        
        return pms.sumrhs(self.mesh, GB) if collapseload else pms.sumleftfaces(self.mesh,GB)
    
    @ppd.parallelmethod()    
    def stiffness(self):
        delta = self.delta
        dik = delta / (1j * self.k)
        B = self.B
        
        # The contribution of the boundary conditions
#        SB = self.bdyassembly.assemble([[(1-delta) * B, (1-delta)*B],
#                              [-dik * B, -dik *B]])
        SB = self.bdyassembly.assemble([[-(1-delta) * B, (1-delta)*B],
                              [dik * B, -dik *B]])
        return pms.sumfaces(self.mesh,SB)     
    
    @ppd.parallelmethod()
    def trace(self):
        ''' projects onto the space spanned by the bdyinfo '''
        return pms.sumleftfaces(self.mesh, )

            

@ppd.distribute()    
class HelmholtzSystem(object):
    ''' Assemble a system to solve a Helmholtz problem using the DG formulation of Gittelson et al.
        It's parallelised so will only assemble the relevant bits of the system for the partition managed by
        this process.
    '''
    def __init__(self, computationinfo, alpha=0.5,beta=0.5,delta=0.5):
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.internalassembly = computationinfo.faceAssembly()
        self.volumeassembly = computationinfo.volumeAssembly()
        self.weightedassembly = computationinfo.volumeAssembly(True)
        self.problem = computationinfo.problem 
        self.computationinfo = computationinfo
        self.boundaries = [self.getBoundary(entity, bdyinfo) for (entity, bdyinfo) in self.problem.bdyinfo.items()]

    def getBoundary(self, entity, bdyinfo):
        return HelmholtzBoundary(self.computationinfo, entity, bdyinfo, self.delta) 

    @ppd.parallelmethod(None, ppd.tuplesum)        
    def getSystem(self, dovolumes = False):
        ''' Returns the stiffness matrix and load vector'''
        SI = self.internalStiffness() 
        SB = sum([b.stiffness() for b in self.boundaries])
        S = SI + SB
        G = sum([b.load() for b in self.boundaries])
        if dovolumes: 
            S+=self.volumeStiffness()
        return S,G

#    @ppd.parallelmethod()        
    def internalStiffness(self):
        ''' The contribution of the internal faces to the stiffness matrix'''
        jk = 1j * self.problem.k
        AJ = pms.AveragesAndJumps(self.problem.mesh)   
        B = self.problem.mesh.boundary # This is the integration by parts term that generally gets folded into the boundary data, but is more appropriate here
        
        SI = self.internalassembly.assemble([[jk * self.alpha * AJ.JD,   -AJ.AN - B], 
                                            [AJ.AD + B,                -(self.beta / jk) * AJ.JN]])    
        SFSI = pms.sumfaces(self.problem.mesh,SI)
                
        return SFSI

#    @ppd.parallelmethod()
    def volumeStiffness(self):
        ''' The contribution of the volume terms to the stiffness matrix (should be zero if using Trefftz basis functions)'''
        E = pms.ElementMatrices(self.problem.mesh)
        L2K = self.weightedassembly.assemble([[E.I, E.Z],[E.Z, E.Z]])
        H1 = self.volumeassembly.assemble([[E.Z,E.Z],[E.Z, E.I]])
        
        AJ = pms.AveragesAndJumps(self.problem.mesh)
        B = pms.sumfaces(self.problem.mesh, self.internalassembly.assemble([[AJ.Z,AJ.Z],[AJ.I,AJ.Z]]))
        
        return H1 - L2K - B   
