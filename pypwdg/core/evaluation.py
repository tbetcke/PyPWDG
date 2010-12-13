'''
Created on Aug 11, 2010

@author: joel
'''

from pypwdg.core.vandermonde import LocalInnerProducts
from pypwdg.utils.geometry import pointsToElementBatch, elementToStructuredPoints
from pypwdg.utils.timing import print_timing
from pypwdg.parallel.decorate import parallelmethod, distribute, tuplesum
from pypwdg.mesh.meshutils import MeshQuadratures

import numpy

@distribute()
class Evaluator(object):
    @print_timing
    def __init__(self, mesh, elttobasis, points):
        self.mesh = mesh
        self.points = points
        ptoe = pointsToElementBatch(points, mesh, 5000)
        # could use sparse matrix classes to speed this up, but it's a bit clearer like this
        # pointsToElement returns -1 for elements which have no point
        self.etop = [[] for e in range(mesh.nelements+1)] 
        for p,e in enumerate(ptoe):
            self.etop[e+1].append(p)
        self.elttobasis = elttobasis        
#        self.v = ElementVandermondes(mesh, elttobasis, lambda e: points[self.etop[e+1]])
    
    @parallelmethod()
    @print_timing
    def evaluate(self, x):
        vals = numpy.zeros(len(self.points), dtype=numpy.complex128)
        for e,p in enumerate(self.etop[1:]):
            v = self.elttobasis.getValues(e, self.points[p])            
            (vidx0,vidx1) = self.elttobasis.getIndices()[e:e+2]
            vals[p] += numpy.dot(v, x[vidx0: vidx1])
        return vals


@distribute()
class StructuredPointsEvaluator(object):
    @print_timing
    def __init__(self, mesh, elttobasis, filter, x):
        self.mesh = mesh
        self.elttobasis = elttobasis
        self.filter = filter
        self.x = x
    
    @parallelmethod(None, lambda (v1,c1), (v2,c2) : (v1+v2, c1+c2))
    @print_timing
    def evaluate(self, structuredpoints):
        vals = numpy.zeros(structuredpoints.length, dtype=numpy.complex128)
        pointcount = numpy.zeros(structuredpoints.length, dtype=int)
        for e in self.mesh.partition:
            pointidxs, points = elementToStructuredPoints(structuredpoints, self.mesh, e)
            if len(pointidxs):
                v = self.elttobasis.getValues(e, points)
                (vidx0,vidx1) = self.elttobasis.getIndices()[e:e+2]
                vals[pointidxs] += numpy.dot(v, self.x[vidx0: vidx1])
                pointcount[pointidxs]+=1
        return self.filter(vals), pointcount


@distribute()
class EvalElementError(object):
    """Object to evaluate the jump of the Dirichlet and Neumann data on the internal interfaces 
       and the boundary error
    """
       
    def __init__(self,mesh,elttobasis, localquads, bnddata,vandermondes, bndvs):
        self.mesh=mesh
        self.v=vandermondes            
        self.bnddata=bnddata
        self.bndvs=bndvs
        self.mqs = MeshQuadratures(mesh, localquads)
        
        self.etoc=elttobasis.getIndices()
        self.facemap=mesh.connectivity*numpy.arange(mesh.nfaces)
        
    @parallelmethod(reduceop=tuplesum)
    def evaluate(self, x):
        """Returns a triple (ed,en,eb), where ed is the jump of the Dirichlet data in the interior,
           en the jump of the Neumann data, and eb the error on the boundary
        """
        
        elem_error_dirichlet=numpy.zeros(self.mesh.nelements)
        elem_error_neumann=numpy.zeros(self.mesh.nelements)
        
        # First do the interior faces
        DD = LocalInnerProducts(self.v.getValues, self.v.getValues, self.mqs.quadweights)
        NN = LocalInnerProducts(self.v.getDerivs, self.v.getDerivs, self.mqs.quadweights)
        
        intdiag=self.mesh.internal.diagonal()
        intfaces=[f for f in range(self.mesh.nfaces) if intdiag[f]==1]
        
        #self.mesh.internal.diagonal()*numpy.arange(self.mesh.nfaces) # List of all interior faces in local partition
        for face1 in intfaces:
            # Get adjacent face
            face2=self.facemap[face1]
            # Compute Inner products
            e1=self.mesh.ftoe[face1]
            e2=self.mesh.ftoe[face2]
            x1=x[self.etoc[e1]:self.etoc[e1+1]]
            x2=x[self.etoc[e2]:self.etoc[e2+1]]
            e11=numpy.dot(numpy.conj(x1),numpy.dot(DD.product(face1,face1),x1))
            e12=numpy.dot(numpy.conj(x1),numpy.dot(DD.product(face1,face2),x2))
            e22=numpy.dot(numpy.conj(x2),numpy.dot(DD.product(face2,face2),x2))
            
            #Add absolute value of errors since floating point errors can make result negative
            #or complex valued 
            elem_error_dirichlet[self.mesh.ftoe[face1]]+=numpy.abs(e11+e22-2*numpy.real(e12))
            
            n11=numpy.dot(numpy.conj(x1),numpy.dot(NN.product(face1,face1),x1))
            n12=numpy.dot(numpy.conj(x1),numpy.dot(NN.product(face1,face2),x2))
            n22=numpy.dot(numpy.conj(x2),numpy.dot(NN.product(face2,face2),x2))
            
            elem_error_neumann[self.mesh.ftoe[face1]]+=numpy.abs(n11+n22+2*numpy.real(n12))
            
        # Now do the boundary faces
        
        elem_error_bnd=numpy.zeros(self.mesh.nelements)
        
        
        for (id, bdycondition), bndv in zip(self.bnddata.items(), self.bndvs):
            lc=bdycondition.l_coeffs
            rc=bdycondition.r_coeffs
            ND=LocalInnerProducts(self.v.getDerivs,self.v.getValues,self.mqs.quadweights)
            GG=LocalInnerProducts(bndv.getValues, bndv.getValues, self.mqs.quadweights)
            GD=LocalInnerProducts(bndv.getValues, self.v.getValues,self.mqs.quadweights)
            GnD=LocalInnerProducts(bndv.getDerivs, self.v.getValues, self.mqs.quadweights)
            GN=LocalInnerProducts(bndv.getValues,self.v.getDerivs,self.mqs.quadweights)
            GnN=LocalInnerProducts(bndv.getDerivs,self.v.getDerivs,self.mqs.quadweights)
            GnGn=LocalInnerProducts(bndv.getDerivs,bndv.getDerivs,self.mqs.quadweights)
            GnG=LocalInnerProducts(bndv.getDerivs,bndv.getValues,self.mqs.quadweights)
            
            facematrix=self.mesh.entityfaces[id]
            facediag=facematrix.diagonal()
            faces=[f for f in range(self.mesh.nfaces) if facediag[f]==1]
            for face in faces:
                e=self.mesh.ftoe[face]
                x1=x[self.etoc[e]:self.etoc[e+1]]
                eDD=numpy.abs(lc[0])**2*numpy.dot(numpy.conj(x1),numpy.dot(DD.product(face,face),x1))
                eNN=numpy.abs(lc[1])**2*numpy.dot(numpy.conj(x1),numpy.dot(NN.product(face,face),x1))
                eGG=numpy.abs(rc[0])**2*numpy.array(GG.product(face,face))
                eGnGn=numpy.abs(rc[1])**2*numpy.array(GnGn.product(face,face))
                    
                eND=2*numpy.real(lc[0]*numpy.conj(lc[1])*numpy.dot(numpy.conj(x1),numpy.dot(ND.product(face,face),x1)))
                eGD=-2*numpy.real(lc[0]*numpy.conj(rc[0])*numpy.dot(GD.product(face,face),x1))
                eGnD=-2*numpy.real(lc[0]*numpy.conj(rc[1])*numpy.dot(GnD.product(face,face),x1))
                eGN=-2*numpy.real(lc[1]*numpy.conj(rc[0])*numpy.dot(GN.product(face,face),x1))
                eGnN=-2*numpy.real(lc[1]*numpy.conj(rc[1])*numpy.dot(GnN.product(face,face),x1))
                eGnG=2*numpy.real(rc[0]*numpy.conj(rc[1])*numpy.array(GnG.product(face,face)))
                
                # Awful hack for different shapes returned for the different scalars
                
                eDD=numpy.squeeze(eDD)
                eNN=numpy.squeeze(eNN)
                eGG=numpy.squeeze(eGG)
                eGnGn=numpy.squeeze(eGnGn)
                eND=numpy.squeeze(eND)
                eGD=numpy.squeeze(eGD)
                eGnD=numpy.squeeze(eGnD)
                eGN=numpy.squeeze(eGN)
                eGnN=numpy.squeeze(eGnN)
                eGnG=numpy.squeeze(eGnG)

                elem_error_bnd[e]+= numpy.abs(eDD+eNN+eGG+eGnGn+eND+eGD+eGnD+eGN+eGnN+eGnG)
                
        return (numpy.sqrt(elem_error_dirichlet),numpy.sqrt(elem_error_neumann), numpy.sqrt(elem_error_bnd))
    
            
              
            
        
        