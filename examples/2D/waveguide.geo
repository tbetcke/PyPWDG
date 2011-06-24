Point(1) = {0, 0,0};
Point(2) = {5, 0,0};
Point(3) = {5, 1,0};
Point(4) = {0, 1,0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line Loop(5) = {1, 2, 3, 4};
Plane Surface(6) = {5};
Physical Line(7) = {3, 1};
Physical Line(8) = {4, 2};
Physical Surface(9) = {6};

Field[1] = Attractor;
Field[1].NodesList = {1,2,3,4};
Field[2] = Threshold;
Field[2].DistMax = .5;
Field[2].DistMin=.2;
Field[2].LcMin=0.05;
Field[2].LcMax=.5;
Field[2].IField =1;

Field[3] = Min;
Field[3].FieldsList = {2};
Background Field = 3;