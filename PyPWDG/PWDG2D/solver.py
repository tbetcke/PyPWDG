def solveDirect(dgstruct):
    """Use direct sparse solver from SciPy to solve the discretized
       DG problem. Depending on the installation UMFPACK or SuperLU are used.
   """
    from scipy.sparse.linalg import spsolve
    print "Solve system of size %d" % dgstruct['nfuns']
    dgstruct['sol']=spsolve(dgstruct['A'],dgstruct['rhs'])
