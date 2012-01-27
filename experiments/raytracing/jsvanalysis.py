'''
Created on Jan 27, 2012

@author: joel
'''
import numpy as np
import matplotlib.pyplot as mp

hankelsquarek20npw11 = ([2, 3, 4, 6, 8, 11, 16], [7.90617325569, 0.141720558712, 0.0171431778165, 0.00117033336744, 0.000183787105603, 3.38537989064e-05, 4.12419940091e-06])
hankelsquarek20npw15 = ([2, 3, 4, 6, 8, 11, 16], [0.121213320687, 0.00260444154676, 0.000229538849713, 9.79278712795e-06, 9.98185797096e-07, 8.18393677137e-08, 4.34740432976e-09])
hankelsquarek20npw21 = ([2, 3, 4, 6, 8, 11, 16], [0.000729729363204, 1.08649973172e-05, 6.94202351295e-07, 1.02152555276e-08, 8.0883780281e-09, 3.9613298757e-05, 0.000130186260208])
hankelsquarek20npw7 = ([2, 3, 4, 6, 8, 11, 16], [4.717686483, 1.39311829211, 1.19964378376, 0.0955515244412, 0.0232481294624, 0.00596034965696, 0.00123434517985])
hankelsquarek20poly0rt = ([2, 3, 4, 6, 8, 11, 16], [0.364470386625, 0.192479110299, 0.134437682598, 0.106819730382, 0.103403378416, 0.105290296514, 0.114882821994])
hankelsquarek20poly1rt = ([2, 3, 4, 6, 8, 11, 16], [0.913678539772, 0.561105342813, 0.338796501421, 0.213675335297, 0.163990417392, 0.218111517226, 0.113529985493])
hankelsquarek20poly2rt = ([2, 3, 4, 6, 8, 11, 16], [0.205230048027, 0.0715593082756, 0.0234474705633, 0.0144524325311, 0.0242302597441, 0.0217734665878, 0.00163954387485])
hankelsquarek20poly3rt = ([2, 3, 4, 6, 8, 11, 16], [0.0832937235836, 0.0355515552855, 0.0272972189035, 0.0086735020989, 0.00118266057347, 0.000166840108392, 3.62667568846e-05])
hankelsquarek20poly4rt = ([2, 3, 4, 6, 8, 11, 16], [0.0516862984436, 0.00917458178046, 0.00143289636526, 0.00013222022314, 3.17306790752e-05, 6.69924356889e-06, 1.06006544886e-06])
hankelsquarek20poly5rt = ([2, 3, 4, 6, 8, 11, 16], [0.0125542356688, 0.000860865014048, 0.000136725441587, 1.26117522512e-05, 2.47738398694e-06, 4.12639539651e-07, 4.69671677895e-08])
hankelsquarek40npw11 = ([2, 3, 4, 6, 8, 11, 16], [7.81443861789, 8.76983363721, 7.77314934263, 0.207332023169, 0.0262885121178, 0.00246689771455, 0.000177966010446])
hankelsquarek40npw15 = ([2, 3, 4, 6, 8, 11, 16], [7.89808961457, 6.98953743751, 0.19527206069, 0.00247661989958, 0.000224311848983, 1.79658032533e-05, 8.64715152409e-07])
hankelsquarek40npw21 = ([2, 3, 4, 6, 8, 11, 16], [2.61480857185, 0.0218094308713, 0.000549572404368, 7.84474139943e-06, 3.61454782789e-07, 1.21608251385e-08, 9.43851121211e-08])
hankelsquarek40npw7 = ([2, 3, 4, 6, 8, 11, 16], [2.09359528647, 13.8281256567, 10.4766727811, 7.43478837026, 3.5879960317, 0.240816309301, 0.0309021813041])
hankelsquarek40poly0rt = ([2, 3, 4, 6, 8, 11, 16,17, 24, 34], [0.481504574492, 0.244552915807, 0.151279320745, 0.0945329046429, 0.0797045593018, 0.0737078881434, 0.0741950395246,0.0744421529888, 0.0779444732249, 0.0823332071032])
hankelsquarek40poly1rt = ([2, 3, 4, 6, 8, 11, 16,17, 24, 34], [0.735699925717, 0.672205240273, 0.506736209359, 0.64295580914, 0.404638457133, 0.358614876677, 0.149684030026,0.127538010854, 0.134221489175, 0.0721244551797])
hankelsquarek40poly2rt = ([2, 3, 4, 6, 8, 11, 16,17, 24, 34], [0.321212829118, 0.201723549886, 0.280965879839, 0.0387709628404, 0.0246990342819, 0.0309401379021, 0.101842658417,0.196138218007, 0.00388940963747, 0.00154327774058])
hankelsquarek40poly3rt = ([2, 3, 4, 6, 8, 11, 16,17, 24, 34], [0.313492150367, 0.103057495881, 0.0412436561026, 0.0461990988353, 0.0175370910681, 0.0415491522805, 0.00478455142747,0.00235271259784, 2.47430289212e-05, 5.22704357602e-06])
hankelsquarek40poly4rt = ([2, 3, 4, 6, 8, 11, 16], [0.152785260288, 0.0372975355507, 0.0127309980281, 0.00382182884796, 0.000470849533052, 0.000134441039957, 3.15829994992e-06])
hankelsquarek40poly5rt = ([2, 3, 4, 6, 8, 11, 16], [0.0828785509245, 0.00371613920181, 0.00220898583645, 0.000121001051608, 1.39362579962e-05, 1.61415953646e-06, 1.71784597946e-07])
hankelsquarek40npw29 = ([2, 3, 4, 6, 8, 11], [0.167970066396, 5.3607536763e-05, 6.05795007004e-07, 4.77202557466e-09, 4.01307121461e-05, 0.440945419905])
hankelsquarek40npw37 = ([2, 3, 4, 6, 8, 11], [1.17180017263, 0.00052381162974, 4.77644484378e-05, 3.88737931578e-06, 0.000434399334444, 1.2574902602])

pw20 = [hankelsquarek20npw7, hankelsquarek20npw11, hankelsquarek20npw15, hankelsquarek20npw21]
ap20 = [hankelsquarek20poly0rt, hankelsquarek20poly1rt, hankelsquarek20poly2rt, hankelsquarek20poly3rt, hankelsquarek20poly4rt, hankelsquarek20poly5rt]
pw40 = [hankelsquarek40npw7, hankelsquarek40npw11, hankelsquarek40npw15, hankelsquarek40npw21, hankelsquarek40npw29, hankelsquarek40npw37]
ap40 = [hankelsquarek40poly0rt, hankelsquarek40poly1rt, hankelsquarek40poly2rt, hankelsquarek40poly3rt, hankelsquarek20poly4rt, hankelsquarek20poly5rt]


markerstyles = ['o-', 's-', '^-', '*-', 'D-', '+-', 'v-']
npws = [7,11,15,21]
npws40 = npws + [29,37]
pdegs = range(6)

def pwdofs(data, npw):
    return ([2* n**2 * npw for n in data[0]], data[1])

def apdofs(data, pdeg):
    return ([n**2* (pdeg)*(pdeg+1) for n in data[0]], data[1])

def loglogplot(l, title = None, names = None):
#    a = np.array(l).transpose((1,2,0))
    mp.figure()
    if names is None: names = range(len(l)) 
    for i, ((N, errs), name) in enumerate(zip(l, names)):
        line = mp.loglog(1.0 / np.array(N), errs, markerstyles[i % len(markerstyles)], label=str(name), basex=2, hold=True)
    if title: mp.suptitle(title)
    mp.legend()
    mp.xlabel('h')
    mp.ylabel('Relative error')

def loglogplot2(l, title, markers):
    mp.figure()
    for (dofs, errs), marker in zip(l, markers):
        mp.loglog(dofs, errs, marker, basex = 10, hold=True) 
    if title: mp.suptitle(title)
        
    mp.xlabel('degrees of freedom')
    mp.ylabel('relative error')

if __name__ == '__main__':
    loglogplot(pw20, 'Plane waves, k=20', npws)
    loglogplot(ap20, "Augmented plane wave, k=20", pdegs)

    loglogplot(pw40, 'Plane waves, k=40', npws40)
    loglogplot(ap40, "Augmented plane wave, k=40", pdegs)
    
    loglogplot2(map(pwdofs, pw20, npws) + map(apdofs, ap20, pdegs), "k = 20", ['bo:']*4 + ['b^-']*6)
    loglogplot2(map(pwdofs, pw40, npws40) + map(apdofs, ap40, pdegs), "k = 40", ['bo:']*6 + ['b^-']*6)
    
    

    mp.show()