load mpg.mat

%n=4, npw = 5

sold = [ 0.1194 +2.8788e-02j  0.1810 +4.9829e-02j  0.0255 -1.9401e-02j -0.0108 -2.0361e-02j  0.0115 -1.8166e-02j  0.1146 +5.4585e-02j  0.1896 +1.3763e-02j  0.0008 -2.1587e-02j -0.0162 +1.0139e-03j  0.0206 -3.2903e-02j  0.0419 +8.4153e-02j  0.2153 +1.4995e-02j  0.0531 -4.3137e-02j  0.0034 -5.1330e-02j  0.0188 +3.7243e-02j  0.0262 +8.8463e-02j  0.2038 -2.5795e-02j  0.0144 -1.8006e-02j  0.0057 -8.7129e-03j  0.0139 +2.1772e-02j -0.0324 +5.8966e-02j  0.2220 -3.4753e-02j  0.0548 -2.8260e-02j  0.0442 +7.7443e-03j -0.0349 -4.1826e-03j -0.0413 +5.1702e-02j  0.2055 -6.5684e-02j  0.0175 -1.5759e-02j  0.0073 +3.6411e-03j -0.0208 +1.8066e-03j -0.0458 +3.3537e-03j  0.2186 -7.8388e-02j  0.0456 -1.9772e-02j -0.0043 +3.1993e-02j  0.0150 -2.0498e-02j -0.0469 -1.6132e-03j  0.2076 -1.0056e-01j  0.0178 -2.7713e-02j  0.0140 +1.1335e-02j  0.0016 -2.2261e-02j  0.1443 -5.0307e-03j  0.1048 +1.1338e-01j  0.0100 +3.1482e-02j  0.0229 -6.1973e-03j  0.0287 -9.6023e-03j  0.1468 +1.2967e-02j  0.1312 +1.1490e-01j 0.0511 +3.2372e-02j -0.0044 -6.0062e-02j  0.0589 -3.0221e-02j  0.0861 +7.7984e-02j  0.1633 +9.0174e-02j  0.0364 +4.4888e-02j  0.0384 -2.5500e-03j  0.0123 +4.2927e-02j  0.0692 +8.6541e-02j  0.1834 +5.9819e-02j  0.0440 +1.6952e-02j  0.0299 -1.6001e-02j  0.0124 +4.3185e-02j -0.0070 +9.0192e-02j  0.1886 +4.2933e-02j  0.0180 +4.8676e-02j -0.0118 +3.1764e-02j -0.0346 -1.3192e-02j -0.0203 +8.5148e-02j  0.2004 +9.2611e-03j  0.0368 +1.9168e-02j  0.0156 +2.3750e-02j -0.0391 -4.9237e-03j -0.0580 +3.5212e-02j  0.2034 -6.1775e-03j  0.0140 +4.0890e-02j -0.0241 -7.4744e-03j  0.0209 -1.8361e-02j -0.0645 +2.9013e-02j  0.2076 -3.7309e-02j  0.0413 +1.7574e-02j -0.0206 +2.6832e-02j  0.0138 -3.4707e-02j  0.1524 -4.3936e-02j  0.0149 +1.3025e-01j -0.0331 +4.7016e-04j  0.0020 +2.5315e-02j  0.0352 +7.4599e-03j  0.1596 -2.9209e-02j  0.0357 +1.4436e-01j -0.0448 +3.6498e-02j  0.0547 +1.5441e-02j  0.0673 +3.6497e-03j  0.1199 +5.4753e-02j  0.0791 +1.4131e-01j -0.0519 +2.1421e-02j -0.0113 +3.6361e-02j -0.0101 +4.5228e-02j  0.1093 +6.8994e-02j  0.1039 +1.2377e-01j -0.0308 +2.8478e-02j -0.0051 +2.7940e-02j -0.0137 +4.3043e-02j  0.0316 +1.0345e-01j  0.1296 +1.1315e-01j -0.0549 +8.9393e-03j -0.0357 -2.1567e-02j -0.0306 -3.0543e-02j  0.0178 +1.0416e-01j  0.1515 +8.9369e-02j -0.0322 +2.8589e-02j -0.0302 +5.2513e-05j -0.0341 -2.5818e-02j -0.0483 +6.8465e-02j  0.1707 +7.0210e-02j -0.0463 +8.8505e-03j  0.0135 -2.9331e-02j  0.0336 -9.7239e-03j -0.0563 +6.5670e-02j  0.1849 +4.0583e-02j -0.0253 +3.7500e-02j -0.0179 -2.6817e-02j  0.0299 -2.5320e-02j  0.1584 -7.7563e-02j -0.0575 +1.1395e-01j -0.0138 -4.4296e-02j -0.0196 +3.3854e-02j  0.0565 +2.4837e-02j  0.1652 -7.1358e-02j -0.0493 +1.1635e-01j -0.0210 -4.6556e-02j -0.0210 +4.7561e-02j  0.0643 +3.0638e-02j  0.1417 +3.3235e-02j  0.0014 +1.5107e-01j -0.0205 -5.2429e-02j -0.0435 -1.2108e-02j -0.0342 +4.6239e-02j  0.1415 +4.5488e-02j  0.0181 +1.3498e-01j -0.0195 -3.1043e-02j -0.0286 -5.7330e-03j -0.0335 +3.8906e-02j  0.0683 +1.0514e-01j 0.0606 +1.4443e-01j -0.0091 -4.9843e-02j  0.0145 -3.6813e-02j -0.0259 -4.3492e-02j  0.0603 +1.1096e-01j  0.0739 +1.3036e-01j -0.0195 -3.3074e-02j -0.0011 -3.0244e-02j -0.0264 -3.9534e-02j -0.0278 +9.6676e-02j  0.1129 +1.2266e-01j -0.0051 -4.6796e-02j  0.0313 +1.1416e-02j  0.0428 +7.7246e-04j -0.0348 +1.0012e-01j  0.1266 +1.0530e-01j -0.0282 -3.4284e-02j  0.0340 -1.4795e-02j  0.0456 -1.2074e-02j];
x = transpose(sum(reshape(sold, 2, []), 1));
max(M \ G - x)

x = M \ G;

o1 = [ 85  86  87  88  89  95  96  97  98  99 105 106 107 108 109 115 116 117 118 119]+1;
o2 = [40 41 42 43 44 50 51 52 53 54 60 61 62 63 64 70 71 72 73 74]+1;

%raw indices:
re2 = [ 80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95  96  97  98  99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119]+1;
re1 = [40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79]+1;

ri1 = [  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39  85  86  87  88  89  95  96  97  98  99 105 106 107 108 109 115 116 117 118 119]+1;
ri2 = [ 40  41  42  43  44  50  51  52  53  54  60  61  62  63  64  70  71  72  73  74 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159]+1;

d1 = o2; % indices of the overlapped dofs in the underlying matrix
d2 = o1;

e1m = setdiff(re1,d1); % external minus overlapped
e2m = setdiff(re2,d2); 
i1m = setdiff(ri1,o1); % internal minus overlapping
i2m = setdiff(ri2,o2);

e1 = [e1m, d1]
e2 = [e2m, d2]
i1 = [i1m, o1]
i2 = [i2m, o2]

p1 = [i1m,e1m,d1,o1]; % the indices of the dofs of the perturbed system from the underlying matrix
p2 = [i2m,e2m,d2,o2];

p = [p1,p2];

ai1m = 1:length(i1m); % where the internal^- dofs sit in the pertubed system
ae1m = (1:length(e1m)) + ai1m(end); % ... external^- ...
ad1 = (1:length(d1)) + ae1m(end); % ... overlapped ...
ao1 = (1:length(o1)) + ad1(end); % etc.

ai2m = (1:length(i2m)) + ao1(end);
ae2m = (1:length(e2m)) + ai2m(end);
ad2 = (1:length(d2)) + ae2m(end);
ao2 = (1:length(o2)) + ad2(end);

ap1 = [ai1m,ae1m,ad1,ao1]; % indices of all the dofs from the first partition in the perturbed system
ap2 = [ai2m,ae2m,ad2,ao2];

ax1 = [ai1m,ae1m,ad1]; % everything except the overlapping dofs (i.e. get rid of the duplicates)
ax2 = [ai2m,ae2m,ad2];
ax = [ax1,ax2];
ux1 = [i1m,e1m,d1]; % and the corresponding dofs from the underlying matrix (i.e. all of them, but ordered)
ux2 = [i2m,e2m,d2];
ux = [ux1,ux2];

P1 = P(d1, o1); % The interesting bit of the perturbation matrix
P2 = P(d2, o2);
%   P1 = rand(size(P1));
%   P2 = rand(size(P2));

n = length(p)
MM = sparse(n,n);
MM(ap1,ap1) = M(p1,p1); % Build the main bit of the perturbed system
MM(ap2,ap2) = M(p2,p2);
MM(ao2, ae1m) = M(o2, e1m);
MM(ao1, ae2m) = M(o1, e2m);

PP = sparse(n,n);
PP(ad1, [ao1, ad2]) = [P1, -P1]; % Build the perturbation
PP(ad2, [ad1, ao2]) = [-P2, P2];


GG = G(p); % some of the RHS needs to be duplicated in order to create the RHS for the perturbed system
q = 1

MP = MM+q * PP

xx = MP \ GG;

max(xx(ad1) - xx(ao2))
max(xx(ad2) - xx(ao1))

erridx = find(abs(xx(ax) - x(ux)) > 1E-4)
x(ux(erridx))
xx(ax(erridx))

% Now do the Schur complement:
e = [e1,e2];

% ni1 = length(i1);
% ni2 = length(i2);
% ne1 = length(e1);
% ne2 = length(e2);
% ne = ne1 + ne2;
% 
% [~,d1ine1] = ismember(d1,e1);
% [~,d2ine2] = ismember(d2,e2);
% [~,e1ine] = ismember(e1,e);
% [~,e2ine] = ismember(e2, e);
% [~,o1ini1] = ismember(o1,i1);
% [~,o2ini2] = ismember(o2,i2);
% 
% 
% 
% Pei1 = sparse(ne1, ni1);
% Pei1(d1ine1,o1ini1) = P1;
% Pei2 = sparse(ne2, ni2);
% Pei2(d2ine2,o2ini2) = P2;
% 
% Pee1 = sparse(ne1, ne);
% Pee1(d1ine1,e2ine(d2ine2)) = P1;
% Pee2 = sparse(ne2, ne);
% Pee2(d2ine2,e1ine(d1ine1)) = P2;


% S1 = M(e1, e)  - (M(e1,i1) + q * Pei1) * (M(i1,i1) \ M(i1,e));
% S2 = M(e2, e)  - (M(e2,i2) + q * Pei2) * (M(i2,i2) \ M(i2,e));
% S1 = M(e1, e) - q * Pee1 - (M(e1,i1) + q * Pei1) * (M(i1,i1) \ M(i1,e));
% S2 = M(e2, e) - q * Pee2 - (M(e2,i2) + q * Pei2) * (M(i2,i2) \ M(i2,e));
S1 = M(e1, e) - q * P(e1,e) - (M(e1,i1) + q * P(e1,i1)) * (M(i1,i1) \ M(i1,e));
S2 = M(e2, e) - q * P(e2,e) - (M(e2,i2) + q * P(e2,i2)) * (M(i2,i2) \ M(i2,e));

g1 = G(e1) - M(e1,i1) * (M(i1,i1) \ G(i1));
g2 = G(e2) - M(e2,i2) * (M(i2,i2) \ G(i2));

xe = [S1;S2] \ [g1;g2];

% Now do the Schur complement blindly:
ae = [ae1m,ad1, ae2m, ad2]; 
ai = [ai1m, ao1, ai2m, ao2];

MPS = MP(ae,ae) - MP(ae,ai) * (MP(ai,ai) \ MP(ai,ae));
GS = GG(ae) - MP(ae,ai) * (MP(ai,ai) \ GG(ai));
xs = MPS \ GS;

[xe,x(e),xs]

