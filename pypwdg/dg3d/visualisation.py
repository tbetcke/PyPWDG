'''
Created on May 28, 2010

@author: joel
'''
from pypwdg.dg3d.utils import *
import numpy
    
class Visualiser(object):
    """ Visualise the results from a Solver"""
    def __init__(self, solver, g):
        from scipy import sparse    
        from numpy import array, ones
        import numpy.matlib
        from enthought.mayavi.mlab import clf
        N = 10
        m = self.mesh = solver.mesh
                    
        refpoints = array([(x*1.0/N,y*1.0/N) for x in range(N,-1,-1) for y in range(N+1-x)])
        
        reftriangles = [(i*(i+1)/2+j, (i+1)*(i+2)/2+j, (i+1)*(i+2)/2+j+1) for i in range(N) for j in range(i+1)]
        reftriangles.extend([(i*(i+1)/2+j, i*(i+1)/2+j+1, (i+1)*(i+2)/2+j+1) for i in range(N) for j in range(i)])
        
        facepoints = m.facePoints(refpoints)
        ud,un = solver.solutionvalues(refpoints)
    
        sfaverage = expandsparse(m.sfaverage, len(refpoints))
        jump = expandsparse(m.jump, len(refpoints))
        boundary = expandsparse(m.boundary, len(refpoints))
        
        self.uaverage = (sfaverage * ud).squeeze()
        
        gv = numpy.vstack(m.values(refpoints, g)).A.squeeze()
        self.gaverage = (sfaverage * gv).squeeze()
        self.ujump = sfaverage * numpy.absolute(jump * ud)
        self.ugjump = numpy.absolute(sfaverage * boundary * (ud - gv))
        
        self.alltriangles = [(x+i,y+i,z+i) for i in range(0,len(self.uaverage), len(refpoints)) for (x,y,z) in reftriangles]
        self.allpoints = numpy.vstack(facepoints).A
        
        self.directions = solver.directions
        self.x = solver.x
     

    def newfig(self):
        from enthought.mayavi.mlab import figure
        figure()
    
    def showmesh(self):
        from enthought.mayavi.mlab import triangular_mesh                 
        m = self.mesh 
        triangular_mesh(m.points[:,0],m.points[:,1],m.points[:,2],numpy.array(m.ftovs),representation="wireframe", scalars = ones(len(m.points)), opacity=0.8)
    
    def display(self, v):
        from enthought.mayavi.mlab import triangular_mesh, colorbar, clf    
        clf()
        self.showmesh()
        UA = triangular_mesh(self.allpoints[:,0], self.allpoints[:,1], self.allpoints[:,2], self.alltriangles, representation="surface", scalars=v, opacity = 0.25)
        colorbar(object=UA, orientation = "vertical")
    
    def showuaveragereal(self):
        self.display(self.uaverage.real)
        
    def showlogerror(self):
        self.display(numpy.log10(numpy.absolute(self.uaverage - self.gaverage))) 
    
    def showjumps(self):
        self.display(numpy.log10(self.ujump + self.ugjump))   
    
    def showdirections(self, n):
        from enthought.mayavi.mlab import quiver3d, vectorbar
        d = self.directions.A
        x,y,z = (d[:,0], d[:,1], d[:,2])
        dn = len(x)
        print dn
        c = 1
#        uabs = numpy.abs(self.x)
        ureal = numpy.abs(numpy.real(self.x))
        uim = numpy.abs(numpy.imag(self.x))
#        space = max(max(ureal[]), max(uim))*1.2
        space = 10.0 #max(uabs)
        for i in range(n):
#            uelt = uabs[i*dn:(i+1)*dn]
#            quiver3d(x*c,y*c+i*space,z*c, x * uelt, y * uelt, z * uelt )
            urelt = ureal[i*dn:(i+1)*dn]
            uielt = uim[i*dn:(i+1)*dn]
            quiver3d(x*c,y*c+i*space,z*c, x * urelt, y * urelt, z * urelt, scale_factor=1.0 )
            quiver3d(x*c,y*c+i*space,z*c+space, x * uielt, y * uielt, z * uielt, scale_factor=1.0 )            
