'''
Created on Aug 9, 2010

@author: joel
'''


def example():
    from PyPWDG.mesh.meshutils import MeshQuadratures
    from PyPWDG.mesh.gmsh_reader import gmsh_reader
    from PyPWDG.mesh.mesh import Mesh
    from PyPWDG.utils.quadrature import legendrequadrature
    from PyPWDG.core.vandermonde import LocalVandermondes, LocalInnerProducts

    mesh_dict=gmsh_reader('/../../examples/2D/square.msh')
    squaremesh=Mesh(mesh_dict,dim=2)
    
    Nq = 5
    qw = MeshQuadratures(squaremesh, legendrequadrature(Nq))
    Np = 8
     