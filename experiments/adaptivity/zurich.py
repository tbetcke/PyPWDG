'''
Created on Nov 3, 2010

@author: joel
'''

import pypwdg.core.adaptivity as pcad
import pypwdg.utils.quadrature as puq
import pypwdg.core.bases as pcb

import math
import numpy as np

def plotsquare3d(us):
    from enthought.mayavi.mlab import surf, show, colorbar, xlabel, ylabel, figure
    for u in us:
        figure()
        Np = 60
        points = uniformsquarepoints(Np)
        x = points[:,0].reshape(Np,Np)
        y = points[:,1].reshape(Np,Np)
        up = np.real(u(points))
        surf(x,y, up.reshape(Np,Np))
        colorbar()
        xlabel('x')
        ylabel('y')
    show()


def plotsquare(u):
    import matplotlib.pyplot as mp
    Np = 30
    points = uniformsquarepoints(Np)
    x = points[:,0].reshape(Np,Np)
    y = points[:,1].reshape(Np,Np)
    up = u(points)
    c = mp.contour(x,y, up.reshape(Np,Np))
    mp.clabel(c)
    mp.show()

def uniformsquarepoints(Np):
    return np.linspace(0,1,Np)[np.mgrid[0:Np,0:Np].reshape(2,-1)].transpose()

def runadaptivebasis2(k, u, Nb, origin, quadrule):
    bases = []
    coefflist = []
    for nfb in range(Nb/2):
        npw = Nb - (2*nfb + 1)
        if npw > 0:
            gen, ini = pcad.pwbasisgeneration(k, npw)
            ini = ini
            fb = pcb.FourierBessel(origin, range(-nfb,nfb+1), k)
            basis, (coeffs, err) = pcad.optimalbasis2(u, gen, ini, quadrule, fb)
            
            l2err = math.sqrt(np.vdot(err,err))
#            print nfb, npw, basis,coeffs,l2err
            print nfb, npw, l2err
            print basis
            print coeffs
            bases.append(basis)
            coefflist.append(coeffs)
    return [pcb.BasisReduce(b, c) for b,c in zip(bases, coefflist)]


def runadaptivebasis(k, u, Nb, origin, quadrule):
    bases = []
    coefflist = []
    for nfb in range(Nb/2):
        npw = Nb - (2*nfb + 1)
        if npw > 0:
            gen, ini = pcad.pwfbbasisgeneration(k, origin, npw, nfb)
            basis, (coeffs, err) = pcad.optimalbasis2(u, gen, ini, quadrule)
            l2err = math.sqrt(np.vdot(err,err))
#            print nfb, npw, basis,coeffs,l2err
            print nfb, npw, l2err
            print basis
            print coeffs
            bases.append(basis)
            coefflist.append(coeffs)
    return pcb.BasisReduce(bases[-1], coefflist[-1])

def runpwbasis(k, u, Npw, quadrule):
    gen, ini = pcad.pwbasisgeneration(k, Npw)
    basis, (coeffs, err) = pcad.optimalbasis2(u, gen, ini, quadrule)
    l2err = math.sqrt(np.vdot(err,err))
    print l2err, basis, coeffs
    return pcb.BasisReduce(basis, coeffs)

if __name__ == '__main__':
    k=15
    origin1 = (-1,-1)
    origin2 = (2,2)
    fh1 = pcb.FourierHankel(origin1, [0], k)
    fh2 = pcb.FourierHankel(origin2, [0], k)
    fh = lambda p : fh1.values(p) + fh2.values(p)
    Npw = 8
    Nq = 15
    quadrule = puq.squarequadrature(Nq)
    runpwbasis(k, fh, Npw, quadrule)
    fs = runadaptivebasis2(k, fh, Npw, [0.5,0.5], quadrule)
    plotsquare3d([f.values for f in fs])
    