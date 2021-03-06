'''
Created on Apr 15, 2011

@author: joel
'''

import numpy as np
import pypwdg.utils.geometry as pug
import pypwdg.output.basis as pob
import pypwdg.core.errors as pce
import pypwdg.output.mploutput as pom

def writeSolutionVTK(solution, bounds, npoints, realdata=True, fname='solution.vti'):
    from pypwdg.output.vtk_output import VTKStructuredPoints

    print "Evaluate Solution and Write to File"
    
    bounds=np.array(bounds,dtype='d')
    filter=np.real if realdata else np.imag

    vtk_structure=VTKStructuredPoints(solution.getEvaluator(filter))
    vtk_structure.create_vtk_structured_points(bounds,npoints)
    vtk_structure.write_to_file(fname)
    
def standardoutput(solution, quadpoints, bounds, npoints, fileroot = None, mploutput = False, **kwargs):
    ''' Dumps the solution to a file and also writes the errors out on a mesh'''
    mesh = solution.computation.problem.mesh
    errors = pce.combinedError(solution)[0]
    print "Combined Error: ",np.sqrt(sum(errors**2))
    volerrors = pce.volumeerrors(solution, quadpoints)
    print "Volume Error / k^2: ", np.sqrt(sum(volerrors **2)) / (solution.computation.problem.k **2)
    if fileroot is not None:
        try:
            writeSolutionVTK(solution, bounds, npoints, fname = fileroot +'.vti')        
            import pypwdg.output.vtk_output as pov
            pov.VTKGrid(mesh, errors).write(fileroot + '.vtu')
        except ImportError as e:   
            print "Some or all output probably failed: ",e
    if mploutput:
        pom.output2dsoln(bounds, solution, npoints,plotmesh=False, **kwargs)
        

def comparetrue(bounds, npoints, g, solution):
    ''' Compare an approximate solution to the true solution
        bounds: bounds for hypercube over which to compare
        npoints: grid to lay on hypercube
        g: true solution
        solution: approximate solution
    '''
    sp = pug.StructuredPoints(bounds.transpose(), npoints)
    idx, points = sp.getPoints(bounds.transpose())
    gvals = np.zeros(sp.length, dtype=complex)
    gvals[idx] = g.values(points)
    l2g = np.sqrt(np.vdot(gvals, gvals) / sp.length)
    perr = solution.evaluate(sp) - gvals
    relerr = np.abs(np.sqrt(np.vdot(perr,perr) / sp.length) / l2g)
    print "Relative L2 error in solution: ", relerr
    return relerr
    


class AdaptiveOutput1(object):
    def __init__(self, computation, quadpoints, bounds, npoints, fileroot, g = None):
        self.computation = computation
        self.quadpoints = quadpoints
        self.bounds = bounds
        self.npoints = npoints        
        self.fileroot = fileroot
        self.g = g
        
    def output(self, i, solution):
        print "Adaptive step ",i
        standardoutput(self.computation, solution, self.quadpoints, self.bounds,self.npoints, "%s%s"%(self.fileroot,i))
        pob.vtkbasis(self.computation.problem.mesh, self.computation.etob, "%s%s%s%s"%(self.fileroot,"dirs", i,".vtu"), solution.x)
        if self.g: comparetrue(self.bounds, self.npoints, self.g, solution)