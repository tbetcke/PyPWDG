'''
Created on Sep 22, 2010

@author: joel
'''
import pypwdg.adaptivity.adaptivity as paa

import numpy as np
#
#def printbasis(elttobasis):
#    for e,bs in enumerate(elttobasis):
#        for b in bs:
#            print b
#            
def vtkbasis(mesh, etob, fname, coeffs):
    ''' Find the directions from a (non-uniform) plane wave basis and output a VTK-compatible file
    
        It's possible that this needs to be updated to work with recent changes to ElementToBasis
    '''
    try:
        import vtk
        
        points = vtk.vtkPoints()
        vectors = vtk.vtkDoubleArray()
        vectors.SetNumberOfComponents(3)
        scalars = vtk.vtkDoubleArray()
        
        nc = 0
        for e in range(mesh.nelements):
            c = paa.origin(mesh, e)
            bs = etob[e]
            cc = np.zeros(3)
            cc[:len(c)] = c
            nondir = 0
            ndir = 0
            for b in bs:
                if hasattr(b, "directions"):
                    for d in b.directions.transpose():
                        dd = np.zeros(3)
                        dd[:len(d)] = d
                        if coeffs is not None: dd *= abs(coeffs[nc])
                        points.InsertNextPoint(*cc)
                        vectors.InsertNextTuple3(*dd)
                        ndir+=1
                        nc+=1
                else:
                    nondir += np.sqrt(np.sum(coeffs[nc:nc+b.n]**2))
                    nc += b.n
            for _ in range(ndir): scalars.InsertNextValue(nondir)
                    
        g = vtk.vtkUnstructuredGrid()
        g.SetPoints(points)
        gpd = g.GetPointData()
        gpd.SetVectors(vectors)
        gpd.SetScalars(scalars)
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(fname)
        writer.SetInput(g)
        writer.Write()
    except ImportError as e:
        print "Unable to write basis to file: ",e
                
                