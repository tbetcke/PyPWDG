'''
Created on Aug 15, 2010

@author: tbetcke
'''

import pypwdg.utils.geometry as pug

import numpy
import vtk

class VTKStructuredPoints(object):
    """ Store output data into a VTK Structured Points object"""
    
    def __init__(self, evaluator):
        self.__evaluator = evaluator
        self.__data = None

    def create_vtk_structured_points(self, boundsd, npointsd):
        bounds = numpy.vstack((boundsd, numpy.zeros((3-len(boundsd),2))))
        npoints = numpy.hstack((npointsd, numpy.ones(3-len(npointsd), dtype=int)))
        
        origin = bounds[:, 0]
        spacing = numpy.hstack((numpy.dot(boundsd, [-1.0,1.0]) / (npointsd-1), numpy.zeros(3-len(boundsd))))
                
        data = vtk.vtkImageData()
        data.SetDimensions(npoints[0], npoints[1], npoints[2])
        data.SetSpacing(spacing[0], spacing[1], spacing[2])
        data.SetNumberOfScalarComponents(1)
        data.SetOrigin(origin[0], origin[1], origin[2])
        data.SetScalarTypeToDouble()
        data.AllocateScalars()
      
        # Create the Data Set
        indices = [[idx, idy, idz] for idx in range(npoints[0]) for idy in range(npoints[1]) for idz in range(npoints[2])]
        points = pug.StructuredPoints(boundsd.transpose(), npointsd)
        vals, counts = self.__evaluator.evaluate(points)
        for i, ind in enumerate(indices):
            data.SetScalarComponentFromDouble(ind[0], ind[1], ind[2], 0, vals[i] / counts[i])
    
        self.__data = data          
        return data #@IndentOk
          
    def write_to_file(self, fname):
        
        if self.__data is not None:
            writer = vtk.vtkXMLImageDataWriter()
            writer.SetFileName(fname)
            writer.SetInput(self.__data)
            writer.Write()
        else:
            print "vtk object does not exist. Use VTKStructuredPoints.create_vtk_structured_points to initialize it"
       
class VTKGrid(object):
    """Store Mesh in a VTK DataType"""
    
    def __init__(self, mesh,scalars=None):
        self.__points = vtk.vtkPoints()
        self.__elems = []
        self.__grid = vtk.vtkUnstructuredGrid()
           
        self.__points.SetNumberOfPoints(mesh.nnodes)
                      
        # Store the points in the VTK Structure
        if mesh.dim == 2:
            for (i, n) in enumerate(mesh.nodes): self.__points.InsertPoint(i, (n[0], n[1], 0.0))
        else:
            for (i, n) in enumerate(mesh.nodes): self.__points.InsertPoint(i, (n[0], n[1], n[2]))
           
        if mesh.dim == 2:
            def create_cell(elem):
                triangle = vtk.vtkTriangle()
                ids = triangle.GetPointIds()
                ids.SetId(0, elem[0])
                ids.SetId(1, elem[1])
                ids.SetId(2, elem[2])
                return triangle
        else:
            def create_cell(elem):
                tetra = vtk.vtkTetra()
                ids = tetra.GetPointIds()
                ids.SetId(0, elem[0])
                ids.SetId(1, elem[1])
                ids.SetId(2, elem[2])
                ids.SetId(3, elem[3])
                return tetra
            
        elements = mesh.elements
        nelems = len(elements)
        self.__elems = [create_cell(elem) for elem in elements]
        self.__grid.Allocate(nelems, 1)
        self.__grid.SetPoints(self.__points)
        for elem in self.__elems:
            self.__grid.InsertNextCell(elem.GetCellType(), elem.GetPointIds())
            
        if scalars is not None:
            pdata=self.__grid.GetCellData()
            data=vtk.vtkDoubleArray()
            data.SetNumberOfValues(nelems)
            for i,p in enumerate(scalars): data.SetValue(i,p)
            pdata.SetScalars(data)
                          
           
           
            
    def write(self, fname):
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(fname)
        writer.SetInput(self.__grid)
        writer.Write()
        
                            
               
                   
        
        
        
    
