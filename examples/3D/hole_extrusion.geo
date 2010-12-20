Point(1) = {0, 0, 0};
Point(2) = {4, 0, 0};
Point(3) = {4, 1, 0};
Point(4) = {0, 1, 0};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Point(5) = {2, .5, 0};
Point(6) = {2.2, .5, 0};
Point(7) = {1.8, .5, 0};
Point(8) = {2, .7, 0};
Point(9) = {2, .3, 0};
Circle(5) = {6, 5, 8};
Circle(6) = {8, 5, 7};
Circle(7)= {7, 5, 9};
Circle(8)= {9, 5, 6};

Line Loop(13) = {1, 2, 3, 4};
Line Loop(14) = {5,6,7,8};
Plane Surface(15) = {13,14};
Extrude {0, 0, .3} {
  Surface{15};
}
Physical Surface(58) = {40};
Physical Surface(59) = {57, 15, 36, 28};
Physical Surface(60) = {32};
Physical Surface(61) = {52, 56, 44, 48};
Physical Volume(62) = {1};

Point(39) = {2, .5, .15};
Field[1] = Attractor;
Field[1].NodesList = {39};
Field[2] = Threshold;
Field[2].DistMax = 0.6;
Field[2].DistMin = 0.4;
Field[2].LcMin = 0.05;
Field[2].IField = 1;

Field[3] = Attractor;
Field[3].NodesList = {1,2,3,4,11,15,10,19};
Field[4]= Threshold;
Field[4].DistMax=0.2;
Field[4].DistMin=0.1;
Field[4].IField=3;
Field[4].LcMin=0.04;
Field[4].LcMax=.5;


Field[5] = Min;
Field[5].FieldsList = {2, 4};
Background Field = 5;
