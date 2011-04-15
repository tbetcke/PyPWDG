'''
Created on Apr 15, 2011

@author: joel
'''

import numpy as np
import pypwdg.utils.geometry as pug

def writeSolutionVTK(solution, bounds, npoints, realdata=True, fname='solution.vti'):
    from pypwdg.output.vtk_output import VTKStructuredPoints

    print "Evaluate Solution and Write to File"
    
    bounds=np.array(bounds,dtype='d')
    filter=np.real if realdata else np.imag

    vtk_structure=VTKStructuredPoints(solution.getEvaluator(filter))
    vtk_structure.create_vtk_structured_points(bounds,npoints)
    vtk_structure.write_to_file(fname)
    
def standardoutput(computation, solution, quadpoints, bounds, npoints, fileroot):
    ''' Dumps the solution to a file and also writes the errors out on a mesh'''
    mesh = computation.problem.mesh
    try:
        writeSolutionVTK(solution, bounds, npoints, fname = fileroot +'.vti')
        
        import pypwdg.core.errors as pce
        import pypwdg.output.vtk_output as pov
        errors = pce.combinedError(computation, quadpoints, solution.x)[0]
        pov.VTKGrid(mesh, errors).write(fileroot + '.vtu')
    except ImportError as e:   
        print "Some or all output probably failed: ",e

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
    print np.sqrt(np.vdot(perr,perr) / sp.length) / l2g
