'''
Created on Nov 18, 2011

@author: joel
'''
import pypwdg.core.bases as pcb
import pypwdg.core.bases.variable as pcbv
import pypwdg.core.bases.reference as pcbr
import pypwdg.core.boundary_data as pcbd
import pypwdg.setup.problem as psp
import pypwdg.setup.computation as psc
import pypwdg.core.physics as pcp
import pypwdg.output.file as pof
import pypwdg.output.mploutput as pom
import pypwdg.test.utils.mesh as tum
import numpy as np
import math
import random
import matplotlib.pyplot as mp
import pypwdg.utils.quadrature as puq
import pypwdg.adaptivity.planewave as pap

class harmonic1():
    ''' Harmonic function s * ((x+t)^2 - (y+t)^2), with s and t chosen such that the gradient has length 1 at (0,0) and self.scale at (1,1)''' 
    
    def __init__(self, scale):
        self.s = (scale - 1) / (2*math.sqrt(2))
        self.t = 1/(2 * math.sqrt(2)*self.s)
        
    def values(self, x):
        return ((x[:,0]+self.t)**2 - (x[:,1]+self.t)**2).reshape(-1,1)*self.s
    def gradient(self, x):
        return (x+[self.t,self.t]) * [2,-2] *self.s

class NormOfGradient():
    def __init__(self, S):
        self.S = S

    def __call__(self, x):
        return np.sqrt(np.sum(self.S.gradient(x)**2, axis=1))

class HarmonicDerived(pcb.Basis):
    def __init__(self, k, S):
        self.k = k
        self.S = S
        
    def values(self, x):
        return np.exp(1j * self.k * self.S.values(x))

    def derivs(self, x, n=None):
        if n is None:
            return (self.S.gradient(x) * self.values(x))[:,np.newaxis,:]
        else:
            return np.dot(self.S.gradient(x), n)[:,np.newaxis] * self.values(x)
    
    def laplacian(self, x):
        return -self.k**2 * self.values(x)

class PlaneWaveFromDirectionsRule(object):
    
    def __init__(self, S, err = 0):
        self.S = S
        self.err = err
        
    def populate(self, einfo):
        import pypwdg.core.bases.definitions as pcbb
        dir = self.S.gradient(einfo.origin)
        M = np.eye(2) + np.array([[0,1],[-1,0]]) * self.err * 2 * (random.random() - 1/2)
        dir = np.dot(M, dir)
        dir = dir / math.sqrt(np.sum(dir**2))    
        return [pcbb.PlaneWaves(dir,einfo.k)]

def variableNhConvergence(Ns, nfn, bdycond, basisrule, process, k = 20, scale = 4.0, pdeg = 1):
    #bnddata={bdytag:pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g=g)}
    bdytag = "BDY"    
    bnddata={bdytag:bdycond}
    entityton ={1:nfn}
    for n in Ns:
        mesh = tum.regularsquaremesh(n, bdytag)
        alpha = ((pdeg*1.0)**2 * n)  / k
        beta = k / (pdeg * 1.0*n) 
        problem=psp.VariableNProblem(entityton, mesh,k, bnddata)
        computation = psc.Computation(problem, basisrule, pcp.HelmholtzSystem, 15, alpha = alpha, beta = beta)
        solution = computation.solution(psc.DirectSolver().solve, dovolumes=True)
        process(n, solution)
    

def genNs(base, minN, maxN):
    a = math.log(minN) / math.log(base)
    b = math.log(maxN) / math.log(base)
    return list(sorted(set(np.array(base**np.arange(a,b),dtype=int))))

def analytichconvergence(maxN, k = 20, scale = 4.0):    
    fileroot = "hconv.k%s.scale%s"%(k,scale)
    bounds=np.array([[0,1],[0,1]],dtype='d')
    npoints=np.array([k * scale * 10,k * scale * 10], dtype=int)
    S = harmonic1(scale)
    g = HarmonicDerived(k, S)   
    nfn = NormOfGradient(S)
    bdycond = pcbd.dirichlet(g)
    
    npw = 15
    pdeg = 2
    Ns = genNs(math.pow(2,1.0/3),1,maxN+1)
    
    pw = pcbv.PlaneWaveVariableN(pcb.circleDirections(npw))
    fo = pof.ErrorFileOutput(fileroot + 'uniformpw%s'%npw, str(Ns), g, bounds, npoints)
    variableNhConvergence(Ns, nfn, bdycond, pw, fo.process, k, scale)
    
    poly = pcbr.ReferenceBasisRule(pcbr.Dubiner(pdeg))
    fo = pof.ErrorFileOutput(fileroot + 'poly%s'%pdeg, str(Ns), g, bounds, npoints)
    variableNhConvergence(Ns, nfn, bdycond, poly, fo.process, k, scale, pdeg)

    for err in [0, 0.02, 0.2]:
        rt = PlaneWaveFromDirectionsRule(S, err)
        fo = pof.ErrorFileOutput(fileroot + 'rt-err%s'%err, str(Ns), g, bounds, npoints)
        variableNhConvergence(Ns, nfn, bdycond, rt, fo.process, k, scale)
        for p in [1,2,3,4]:
            poly = pcbr.ReferenceBasisRule(pcbr.Dubiner(p))
            polyrt = pcb.ProductBasisRule(poly, rt)
            fo = pof.ErrorFileOutput(fileroot + 'poly%srt-err%s'%(p,err), str(Ns), g, bounds, npoints)
            variableNhConvergence(Ns, nfn, bdycond, polyrt, fo.process, k, scale, p)

def analyticconvergencepwprod(maxN, k = 20, scale = 4.0):
    fileroot = "hconv.k%s.scale%s"%(k,scale)
    bounds=np.array([[0,1],[0,1]],dtype='d')
    npoints=np.array([k * scale * 10,k * scale * 10], dtype=int)
    S = harmonic1(scale)
    g = HarmonicDerived(k, S)   
    nfn = NormOfGradient(S)
    bdycond = pcbd.dirichlet(g)
    
    npw = 15
    Ns = genNs(math.pow(2,1.0/2),1,maxN+1)

    
    pw = pcbv.PlaneWaveVariableN(pcb.circleDirections(npw))
    
    for p in [1,2,3,4]:
        poly = pcbr.ReferenceBasisRule(pcbr.Dubiner(p))
        polypw = pcb.ProductBasisRule(poly, pw)
        fo = FileOutput(fileroot + 'pw%spoly%s'%(npw,p), str(Ns), g, bounds, npoints)
        variableNhConvergence(Ns, nfn, bdycond, polypw, fo.process, k, scale, p)

def showtruesoln(k, scale):
    S = harmonic1(scale)
    g = HarmonicDerived(k, S)   
    bounds=np.array([[0,1],[0,1]],dtype='d')
    npoints=np.array([k * scale * 10,k * scale * 10], dtype=int)
    pom.output2dfn(bounds, g.values, npoints)

hconvk20scale40poly1rterr002 = ([1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50], [0.990134706309, 1.05262217864, 0.937071087439, 0.680171598443, 0.53491843574, 0.460905905757, 0.334503099871, 0.243298428851, 0.186742566223, 0.11660495019, 0.0774025297182, 0.0491654196094, 0.0289144505415, 0.0176468728289, 0.0109022093342])
hconvk20scale40poly1rterr02 = ([1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50], [1.00324352398, 1.0463753955, 1.60581930782, 1.07948574187, 0.924566715094, 0.956003025167, 0.797831033272, 0.738534492268, 0.585847929576, 0.371044638417, 0.281812590397, 0.184514467549, 0.12905919196, 0.0952987061716, 0.0548803015932])
hconvk20scale40poly1rterr0 = ([1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50], [0.98753655758, 1.04135683338, 0.764255244844, 0.603142774464, 0.484673373214, 0.446875166013, 0.341107762155, 0.250425835404, 0.190452054733, 0.117522335931, 0.077060809454, 0.0487046479495, 0.0284843634986, 0.0173710173484, 0.0106597480185])
hconvk20scale40poly2rterr002 = ([1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50], [0.976205138134, 0.730663287805, 0.213852222733, 0.0729209750334, 0.0382815710735, 0.0256973396921, 0.00838350214493, 0.00486344300276, 0.0025483325655, 0.000956712343269, 0.000405317768127, 0.000212786437357, 8.56105752917e-05, 5.14494256693e-05, 2.552408471e-05])
hconvk20scale40poly2rterr02 = ([1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50], [0.997959066238, 0.983701130387, 1.29432033402, 0.91458823488, 0.765712066949, 0.513109035208, 0.272662057042, 0.102009785056, 0.0570050177087, 0.0317008769693, 0.0135608411693, 0.00773774163181, 0.00357030513949, 0.00156467152006, 0.000813375817888])
hconvk20scale40poly2rterr0 = ([1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50], [0.974885803395, 0.69959343886, 0.22831859253, 0.0752785763548, 0.028869606135, 0.0143091289659, 0.00494757158233, 0.002155349266, 0.00106193918007, 0.000324096492531, 0.000121571959118, 4.3254933349e-05, 1.30315149327e-05, 4.30346657715e-06, 1.52603251304e-06])
hconvk20scale40poly2 = ([1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50], [1.00008385625, 1.00014398887, 0.999585536679, 0.998377893009, 0.99676498867, 0.993138009624, 0.971852344697, 0.94924677368, 0.923809429221, 0.769653731689, 0.555114152289, 0.33091392433, 0.162165281995, 0.0813306783309, 0.0457561011474])
hconvk20scale40poly3rterr002 = ([1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50], [1.04705113306, 2.35322119719, 0.157813185253, 0.0544137716845, 0.0256047037965, 0.0133733350705, 0.00464665598252, 0.00199957225077, 0.000949458775336, 0.000226531966329, 6.54880195344e-05, 2.36098171128e-05, 1.02343017728e-05, 4.68319766933e-06, 1.71298915095e-06])
hconvk20scale40poly3rterr02 = ([1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50], [1.00768929213, 1.15746167002, 1.00892087646, 0.482567244526, 0.20096100093, 0.107487496898, 0.0350848001122, 0.023260275597, 0.011006761732, 0.00348185892707, 0.00122019498407, 0.000554789831064, 0.000271501185311, 0.000235717893493, 6.63816339815e-05])
hconvk20scale40poly3rterr0 = ([1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50], [1.04528975847, 2.96961225828, 0.150592899707, 0.053259784683, 0.0246951367263, 0.012924210063, 0.00450428547165, 0.0019281426957, 0.000913008029237, 0.000222423319391, 6.64159401701e-05, 2.38278752127e-05, 1.00504274861e-05, 4.41751508331e-06, 1.64526886727e-06])
hconvk20scale40poly4rterr002 = ([1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50], [1.05019642149, 0.599196660871, 0.0190100623618, 0.00240407274679, 0.00120459824962, 0.000418628803777, 0.000108896868542, 4.17715423241e-05, 1.96334974731e-05, 3.99783248835e-06, 1.42575589981e-06, 3.61735614863e-07, 8.27559077141e-08, 2.75919170924e-08, 7.86530770911e-09])
hconvk20scale40poly4rterr02 = ([1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50], [1.04711926199, 1.95175461699, 0.401628114828, 0.262590318106, 0.065617205962, 0.0461095580628, 0.0100502428446, 0.00395972700571, 0.00189814837434, 0.000616441348247, 0.000165991835524, 7.7957259158e-05, 1.51857226577e-05, 3.3709030862e-06, 1.11905792471e-06])
hconvk20scale40poly4rterr0 = ([1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50], [1.04979653942, 0.819690964678, 0.0186583729743, 0.00247255617253, 0.000530160158474, 0.000168406954399, 3.16497474569e-05, 8.59304790816e-06, 2.74390579848e-06, 2.99130874698e-07, 6.55892751452e-08, 1.565790894e-08, 3.20879136277e-09, 8.2230129373e-10, 2.13071268398e-10])
hconvk20scale40pw15poly1 = ([1, 2, 4, 5, 8, 11], [1.05403903891, 4.27943407822, 3.73460277896, 0.873682377824, 0.0536792908222, 1.31189523758])
hconvk20scale40pw15poly2 = ([1, 2, 4, 5, 8, 11], [2.2575564423, 1.75730709091, 0.0102135957155, 0.00788050486234, 1.1821313077, 4552.76781462])
hconvk20scale40pw15poly3 = ([1, 2, 4, 5, 8, 11], [2.96891621478, 50.6491221044, 0.0252128587325, 0.00115313928159, 0.458216319101, 1689.04899069])
hconvk20scale40pw15poly4 = ([1, 2, 4, 5, 8, 11], [12.2608821155, 3.15350790329, 0.118485906914, 0.0124662246762, 4.45432184491, 3983.50706454])
hconvk20scale40rterr002 = ([1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50], [1.0001043193, 1.00017750614, 0.998630613191, 0.987917354346, 0.955765821728, 0.848112922018, 0.601309411766, 0.393697422222, 0.312124244275, 0.22265233838, 0.175338670594, 0.126756653839, 0.103385801527, 0.0815626974211, 0.0668645427897])
hconvk20scale40rterr02 = ([1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50], [1.00076563906, 0.999777732828, 1.00092207881, 1.0065543552, 1.0013874096, 1.00179430855, 0.984148385686, 0.973544690198, 0.95066407368, 0.910909219942, 0.891252694745, 0.859512561624, 0.814227002406, 0.777389296345, 0.747409009581])
hconvk20scale40rterr0 = ([1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50], [0.999987763459, 0.999998688602, 0.998695021508, 0.988736948283, 0.947937400146, 0.849097798813, 0.54776854266, 0.335158003765, 0.229946465055, 0.140551524422, 0.100425337716, 0.0725510836858, 0.0508355763001, 0.0369725138659, 0.0269970597301])
hconvk20scale40uniformpw15 = ([1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50], [1.01681168153, 1.10705077232, 1.33438531356, 1.46476710613, 1.19990671841, 0.998644027297, 0.428673767125, 0.373849778453, 0.215334194475, 0.102780031159, 0.0885663069979, 0.0406074600747, 0.0170051405272, 0.00828574487183, 0.00407856098455])

def loglogplot(l, title = None, names = None):
    a = np.array(l).transpose((1,2,0))
    mp.figure()
    lines = mp.loglog(1.0 / a[0], a[1])
    if title: mp.suptitle(title)
    if len(lines) > 1:
        if names is None: names = range(len(lines)) 
        mp.figlegend(lines, names, 'right')
    
def plotanalytic():
    loglogplot([hconvk20scale40uniformpw15], 'Uniform plane waves (15)', None)
    loglogplot([hconvk20scale40poly2], 'Polynomials of degree 2', None)
    loglogplot([hconvk20scale40rterr0, hconvk20scale40poly1rterr0, hconvk20scale40poly2rterr0, hconvk20scale40poly3rterr0, hconvk20scale40poly4rterr0], 'Ray-traced directions augmented with polynomials of degree p', ['p=0','p=1','p=2','p=3','p=4'])
    loglogplot([hconvk20scale40rterr02, hconvk20scale40poly1rterr02, hconvk20scale40poly2rterr02, hconvk20scale40poly3rterr02, hconvk20scale40poly4rterr02], 'Ray-traced directions with 10% error, augmented with polynomials of degree p', ['p=0','p=1','p=2','p=3','p=4'])
    loglogplot([hconvk20scale40rterr002, hconvk20scale40poly1rterr002, hconvk20scale40poly2rterr002, hconvk20scale40poly3rterr002, hconvk20scale40poly4rterr002], 'Ray-traced directions with 1% error, augmented with polynomials of degree p', ['p=0','p=1','p=2','p=3','p=4'])
    loglogplot([hconvk20scale40poly2rterr0, hconvk20scale40poly2rterr002, hconvk20scale40poly2rterr02],'Ray-traced directions with errors, augmented with polynomials of degree 2', ['0%', '1%', '10%'])
    loglogplot([hconvk20scale40pw15poly1, hconvk20scale40pw15poly2, hconvk20scale40pw15poly3, hconvk20scale40pw15poly4], 'Polynomial convergence', ['p=1','p=2','p=3','p=4'])

class GaussianBubble:
    def __init__(self, c = 1, O = [0.5,0.3]):
        self.c = c
        self.O = O
    
    def __call__(self,x):
        r2 = np.sum((x - self.O)**2, axis=1)                
        return 1.0 / ((1- np.exp(-32*r2)/2) * self.c)


def pwproduniform(g, qxw, k, n):
    theta = np.linspace(0, 2*math.pi, n, endpoint=False)
    return (theta,)+ pap.L2Prod(g, qxw, k).products(theta)

    
def microlocal():
    N = 20
    k = 20    
    qxw = puq.squarequadrature(N)
    g = pcb.BasisReduce(pcb.BasisCombine([pcb.FourierHankel([-1,-0.5], [0], k), pcb.FourierHankel([-0.2,0.5], [0], k)]), [1,1])
    theta, proj, projd, projdd = pwproduniform(g.values, qxw, k, 500)
    mp.plot(theta, proj[0])
    

import pypwdg.parallel.main



if __name__ == '__main__':
    pass    
#    analytichconvergence(60)
#    analyticconvergencepwprod(12)
    #showtruesoln(20,4.0)
    
    #plotanalytic()
    #microlocal()
    
    
    
    