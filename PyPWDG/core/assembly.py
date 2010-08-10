'''
Created on Aug 9, 2010

@author: joel
'''



from PyPWDG.mesh.meshutils import MeshQuadratures
from PyPWDG.mesh.gmsh_reader import gmsh_reader
from PyPWDG.mesh.mesh import Mesh
from PyPWDG.utils.quadrature import legendrequadrature
from PyPWDG.core.vandermonde import LocalVandermondes, LocalInnerProducts
from PyPWDG.core.bases import circleDirections, PlaneWaves
from PyPWDG.mesh.structure import StructureMatrices
from PyPWDG.utils.sparse import createvbsr

mesh_dict=gmsh_reader('../../examples/2D/square.msh')
squaremesh=Mesh(mesh_dict,dim=2)
print squaremesh.etof

SM = StructureMatrices(squaremesh)

k = 3
jk = 1j * k
jki = 1/jk
alpha = 1.0/2
beta = 1.0/2
delta = 1.0/2

Nq = 5
qw = MeshQuadratures(squaremesh, legendrequadrature(Nq))
Np = 9
dirs = circleDirections(Np)
pq = PlaneWaves(dirs, k)
elts = mesh_dict['elements']
elttobasis = dict([(e, pq) for e in elts])
numbases = [e.n for b in elts]


lv = LocalVandermondes(squaremesh, elttobasis, qw.quadpoints)

DD = LocalInnerProducts(lv.getValues, lv.getValues, qw.quadweights)
DN = LocalInnerProducts(lv.getValues, lv.getDerivs, qw.quadweights)
ND = LocalInnerProducts(lv.getDerivs, lv.getValues, qw.quadweights)
NN = LocalInnerProducts(lv.getDerivs, lv.getDerivs, qw.quadweights)

DDI = createvbsr(SM.internal, DD, numbases, numbases) 
DNI = createvbsr(SM.internal, DN, numbases, numbases) 
NDI = createvbsr(SM.internal, ND, numbases, numbases) 
NNI = createvbsr(SM.internal, NN, numbases, numbases)

DDB = createvbsr(SM.boundary, DD, numbases, numbases) 
DNB = createvbsr(SM.boundary, DN, numbases, numbases) 
NDB = createvbsr(SM.boundary, ND, numbases, numbases) 
NNB = createvbsr(SM.boundary, NN, numbases, numbases)

SI = jk * alpha * DDI * SM.JD - DNI * SM.AN + NDI * SM.AD - beta * jki * NNI * SM.JN
SB = jk * (1-delta) * DDB - delta * DNB + (1-delta) * NDB - delta * jki * NNB

S = SM.sumfaces(SI + SB) 
 
    
    