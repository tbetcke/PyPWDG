'''
Created on 3 Mar 2011

@author: jphillips
'''
import pypwdg.adaptivity.adaptivity as paa
import pypwdg.output.basis as pob
import pypwdg.utils.geometry as pug

import numpy as np

def referencesolution(bounds, npoints, g):
    sp = pug.StructuredPoints(bounds.transpose(), npoints)
    idx, points = sp.getPoints(bounds.transpose())
    gvals = np.zeros(sp.length, dtype=complex)
    gvals[idx] = g.values(points)
    l2g = np.sqrt(np.vdot(gvals, gvals) / sp.length)
    return sp, l2g, gvals



def runadaptive(problem, ibc, name, nits, bounds, npoints, refg = None):
    if refg:
        sp,l2g, gvals = referencesolution(bounds, npoints, refg)
        
    def output(i, sol):
        err = sol.combinedError()
        errl2 = np.sqrt(sum(err**2))
        print i, errl2
        if refg:
            perr = sol.evaluate(sp) - gvals
            perrl2 = np.sqrt(np.vdot(perr,perr) / sp.length) / l2g
            print i, perrl2
        sol.writeSolution(bounds,npoints,fname='%s%s.vti'%(name,i))
        pob.vtkbasis(problem.mesh, comp.etob, "%sdirs%s.vtu"%(name,i), sol.x)
        return (errl2, perrl2) if refg else errl2    
        
    comp = paa.AdaptiveComputation(problem, ibc)
    solution = comp.solve()
    stats = []
    stats.append(output(0, solution))
    for i in range(1,nits):
        comp.adapt()
        sol = comp.solve()
        stats.append(output(i,sol))
    print stats