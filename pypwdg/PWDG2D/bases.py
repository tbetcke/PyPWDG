import numpy

class PlaneWaveBasis(object):
    """A planewave basis class

        Example:
        b = PlaneWaveBasis(k=10,p=5) assigns each element 5 planewave
        directions with k=10
    """
    def __init__(self,**kwargs):
        self.k=kwargs['k']
        self.p=kwargs['p']
    
    def eval(self,x,y):
        """ Evaluate the planewave basis

        Arguments:
        x - Array of x-coordinates
        y - Array of y-coordinates

        Return:

        Matrix of function values

        f(x1,y1,p1),     f(x1,y1,p2),   ... , f(x1,y1,p_n)
        ...
        f(x_m,y_m,p1),   f(x_m,y_m,p1), ... , f(x_m,y_m,p_n)

        where the p_j are the planewave directions

        """
        angles=2*numpy.pi*numpy.arange(self.p)/self.p
        directions=numpy.array([numpy.cos(angles),numpy.sin(angles)])
        args=1j*numpy.dot(numpy.array([x,y],dtype='c16').transpose(),directions)*self.k
        return numpy.exp(args)

    def evalNormal(self,x,y,n):
        """Evaluate normal derivates of a planewave basis

        Arguments:
        x - Array of x-coordinates
        y - Array of y-coordinates
        n - numpy matrix [[nx1,ny1],[nx2,ny2],..,[nx_m,ny_m]] of
            m normal directions, where m=1 or same length as coordinates
            If m=1 the same normal direction is used for all arguments

        Return:

        Matrix of function values

        f(x1,y1,p1),     f(x1,y1,p2),   ... , f(x1,y1,p_n)
        ...
        f(x_m,y_m,p1),   f(x_m,y_m,p1), ... , f(x_m,y_m,p_n)

        where the p_j are the planewave directions


        """
        angles=2*numpy.pi*numpy.arange(self.p)/self.p
        directions=numpy.array([numpy.cos(angles),numpy.sin(angles)])
        args=1j*numpy.dot(numpy.array([x,y],dtype='c16').transpose(),directions)*self.k
        return numpy.dot(n,directions)*1j*self.k*numpy.exp(args)   

if __name__=="__main__":

    base=PlaneWaveBasis(k=10,p=5)
    print base.eval([1,2],[3,4])
    print base.evalNormal([1,2],[3,4],numpy.array([1,1])*1/numpy.sqrt(2))
    
