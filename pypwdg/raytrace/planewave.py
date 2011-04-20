import numpy as np
import pypwdg.core.bases as pcb
import math

def absderivs(f, df, ddf, scale = 1):
    ''' given f, df and d^2 f, return abs(f)^2, d (abs(f)^2) and d^2 (abs(f)^2)'''
    f2 = np.abs(f * f.conj())
    df2 = 2 * np.real(df * f.conj())
    ddf2 = 2 * np.real(ddf* f.conj() + df * df.conj())
    return f2 * scale, df2 * scale, ddf2 * scale

class L2Prod(object):
    def __init__(self, g, qxw, k):
        self.qx, qw = qxw    
        gv = g.values(self.qx).reshape(-1,1) 
        self.k = k
        self.gvqw = gv * qw.reshape(-1,1)    
        self.pwnorm2 = sum(qw)
        self.gnorm2 = np.abs(np.dot(self.gvqw.conj().T,  gv)) 
        
    
    def products(self, thetas):
        dirs = np.vstack([np.cos(thetas), np.sin(thetas)])
        ddirs = np.vstack([-dirs[1], dirs[0]])
        
        pw = pcb.PlaneWaves(dirs.T, self.k)
        pwv = pw.values(self.qx)
        
        qxdd = np.dot(self.qx,ddirs)
        qxd = np.dot(self.qx, dirs)
                
        ip = np.dot(self.gvqw.conj().T,pwv)
        ipd = np.dot(self.gvqw.conj().T,  pwv * 1j * self.k * qxdd)
        ipdd = np.dot(self.gvqw.conj().T, pwv * (-1j * self.k * qxd - self.k**2 * qxdd*qxdd.conj()))
        return absderivs(ip ,ipd, ipdd, 1/(self.gnorm2 * self.pwnorm2))
#                   
#class ImpedanceProd(object):
#    def __init__(self, bc, qxw, normal, k):
#        self.qx, qw = qxw    
#        self.k = k
#        self.normal = normal
#        rc0, rc1 = bc.r_coeffs
#        gv = (bc.values(self.qx) * rc0 + bc.derivs(self.qx, normal) * rc1).reshape(-1,1)
#        self.lc = bc.l_coeffs
#        self.gvqw = gv * qw.reshape(-1,1)    
#        self.gnorm2 = np.dot(self.gvqw.conj().T,  gv) 
#        self.pwnorm2 = sum(qw) * (self.lc[0]**2 + self.lc[1]**2)
#        
#    def products(self, thetas):
#        dirs = np.vstack([np.cos(thetas), np.sin(thetas)])
#        ddirs = np.vstack([-dirs[1], dirs[0]])
#                
#        qxd = np.dot(self.qx, dirs)
#        qxdd = np.dot(self.qx,ddirs)
#
#        nd = np.dot(self.normal, dirs)
#        ndd = np.dot(self.normal, ddirs)
#        
#        pw = pcb.PlaneWaves(dirs.T, self.k).values(self.qx)
##        pwn = pw * 1j * self.k * nd
#        
#        lc0,lc1 = self.lc
##        pwi = lc0 * pw + lc1 * pwn
##        
##        pwinorm2 = np.sum(pwi * pwi.conj() * self.qw.reshape(-1,1), axis=0)
##        print pwinorm2
#        
#        # (derivatives of) inner products of pw with g
#        ip = np.dot(self.gvqw.conj().T,pw)
#        ipd = np.dot(self.gvqw.conj().T,  pw * 1j * self.k * qxdd)
#        ipdd = np.dot(self.gvqw.conj().T, pw * (-1j * self.k * qxd - self.k**2 * qxdd*qxdd.conj()))
#        
#        # (derivatives of) inner products of d_n pw with g
#        ipn = ip * 1j * self.k * nd
#        ipnd = (ipd * nd + ip * ndd) * 1j * self.k 
#        ipndd = (ipdd * nd + 2 * ipd * ndd - ip * nd) * 1j * self.k
#        
##        print np.vstack((ip,ipd,ipdd,ipn,ipnd,ipndd))
##        print np.vstack((lc0 * ip + lc1 * ipn,lc0 * ipd + lc1 * ipnd,lc0 * ipdd + lc1 * ipndd))
#
#        return absderivs(lc0 * ip + lc1 * ipn,lc0 * ipd + lc1 * ipnd,lc0 * ipdd + lc1 * ipndd, 1/(np.abs(self.gnorm2 +self.pwnorm2)) )
#        
def findpw(ips, diameter, threshold = 0.2, maxtheta = 0):
    n = 2 * int(ips.k*diameter) # this is a bit of a guess right now 
    theta = np.linspace(0, 2*math.pi, n, endpoint = False)
    thresh2 = threshold**2
    for i in range(5):
        f, fd, fdd = ips.products(theta)
        theta = theta - fd[0] / fdd[0]
#        print i, np.vstack((theta, f[0], fd[0], fdd[0])).T
        notclose = np.abs(np.fmod((np.roll(theta,-1) - theta), 2*math.pi)) > (math.pi / (10 *n)) if len(theta) > 1 else theta==theta
        if len(theta) == 2 and notclose.any(): notclose = [True, False]
        notsmall = f[0] >= thresh2 
        notmin = fdd[0] < 0
        theta = np.sort(np.fmod(theta[notsmall * notclose * notmin] + 2 * math.pi, 2*math.pi))
        if len(theta) ==0: break
#        thresh2 = max(thresh2, np.max(f[0])/2)
#        print thresh2
    if maxtheta:
        f, fd, fdd = ips.products(theta)
        idx = np.argsort(f[0])
        return theta[idx][-maxtheta:]
    else:
        return theta

def findpwds(*args, **kwargs):
    thetas = findpw(*args, **kwargs)
    return np.vstack([np.cos(thetas), np.sin(thetas)])
    
