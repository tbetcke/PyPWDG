'''
Created on Mar 22, 2011

@author: joel
'''
import pypwdg.raytrace.planewave as prp
import numpy as np

def starttracing(mesh, bdydata, k, mqs, maxspace, dotrace):
    for (bdy, bc) in bdydata.items():
        faces = mesh.entityfaces[bdy]            
        for f in faces.tocsr().indices:            
            qp = mqs.quadpoints(f)
            qw = mqs.quadweights(f)
            thetas = prp.findpw(prp.L2Prod(bc, (qp,qw), k), 2, maxtheta = 2)
            dirs = np.vstack([np.cos(thetas), np.sin(thetas)]).T
            n = mesh.normals[f]
            ips = -np.vdot(dirs, n)
            for dir, ip in zip(dirs, ips):
                if ip > 0 :
                    intervals = np.ceil(ip / maxspace)+2
                    refp = np.linspace(0,1,intervals)[1:-1]
                    facedirs = mesh.directions[f]
                    facep = facedirs[0] + np.dot(facedirs.reshape(-1,1), dirs[[1]])
                    for p in facep:
                        dotrace(p, dir, f)
                    
            