'''
Created on Aug 9, 2010

@author: joel
'''

import numpy

from pypwdg.core.vandermonde import LocalInnerProducts
from pypwdg.utils.sparse import createvbsr
from pypwdg.utils.timing import print_timing

class Assembly(object):
    """ Assemble a global matrix based on local inner products and structure matrices 
    
    We suppose that we want to assemble the matrix:
    
    diag(U_1, ... U_n)^H * S^t * diag(W_1 ... W_n) * T * diag(V_1 ... V_n)
    
    where the matrices satisfy the following conditions:
    - the number of rows in U_i = b_i = number of rows in V_i;
    - S and T have a block structure where the entries T_ij = S_ij = zeros(b_i, b_j) if b_i <> b_j. 
      If b_i = b_j then S_ij = s_ij * I and T_ij = t_ij * I where I is a b_i x b_i identity matrix.
    - W_i is a b_i x b_i diagonal matrix (and if any S_ij or T_ij is non-zero then W_i = W_j).
    
    Think: The block structure is at the face level; b_i is the number of quadrature points on each face; 
    U and V are Vandermonde matrices; S and T are structure matrices, e.g. representing averages or jumps 
    across faces; and W_i are the quadrature weights on each face.
    
    We can calculate the matrix by first calculating r = s^H * t and then taking each block as
    r_ij U_i^H * W_i * V_j.  That's what this class facilitates.  
    
    U and V are supplied as LocalVandermonde objects.  These supply values and normal derivatives 
    
    LocalInnerProduct objects are used to manage the blocks U_i^H * W_i * V_j for each of the 4 combinations of 
    values and normal derivatives.
    
    The assemble method accepts a 2x2 array of structure matrices, r.   
    """
    def __init__(self, lv, rv, qws):
        """ 
            lv: Left Vandermonde object
            rv: Right Vandermonde object
            qws: Callable giving quadrature weights for each face
        """
        DD = LocalInnerProducts(lv.getValues, rv.getValues, qws)
        DN = LocalInnerProducts(lv.getValues, rv.getDerivs, qws)
        ND = LocalInnerProducts(lv.getDerivs, rv.getValues, qws)
        NN = LocalInnerProducts(lv.getDerivs, rv.getDerivs, qws)
        
        self.ips = numpy.array([[DD,DN],[ND,NN]])
        self.numleft = lv.numbases
        self.numright = rv.numbases
    
#    @print_timing    
    def assemble(self, structures):
        """ Given a 2x2 array of structure matrices, return an assembled variable block sparse matrix
        
            A structure matrix SM.JD, for example, corresponds to the product u * [[v]].  
            SM.JD^T * SM.JD, would correspond to [[u]] * [[v]] 
        """
        return sum([createvbsr(structures[i,j],self.ips[i,j].product, self.numleft, self.numright) for i in [0,1] for j in [0,1]])
            

