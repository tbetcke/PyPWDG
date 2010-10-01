Point(1) = {3, 0, 0};
Point(2) = {0, 0, 0};
Point(3) = {0, 0, 3 };
Point(4) = {-3,0,0};
Point(5) = {0,3,0};
Circle(1) = {1, 2, 3};
Extrude {{1, 0, 0}, {0, 0, 0}, Pi/4} { Line{1};}
Extrude {{1, 0, 0}, {0, 0, 0}, Pi/4} { Line{2};}
Extrude {{1, 0, 0}, {0, 0, 0}, Pi/4} { Line{5};}
Extrude {{1, 0, 0}, {0, 0, 0}, Pi/4} { Line{8};}
Extrude {{1, 0, 0}, {0, 0, 0}, Pi/4} { Line{11};}
Extrude {{1, 0, 0}, {0, 0, 0}, Pi/4} { Line{14};}
Extrude {{1, 0, 0}, {0, 0, 0}, Pi/4} { Line{17};}
Extrude {{1, 0, 0}, {0, 0, 0}, Pi/4} { Line{20};}
Circle(30)= {5,2,4};
Extrude {{1, 0, 0}, {0, 0, 0}, Pi/4} { Line{30};}
Extrude {{1, 0, 0}, {0, 0, 0}, Pi/4} { Line{31};}
Extrude {{1, 0, 0}, {0, 0, 0}, Pi/4} { Line{34};}
Extrude {{1, 0, 0}, {0, 0, 0}, Pi/4} { Line{37};}
Extrude {{1, 0, 0}, {0, 0, 0}, Pi/4} { Line{40};}
Extrude {{1, 0, 0}, {0, 0, 0}, Pi/4} { Line{43};}
Extrude {{1, 0, 0}, {0, 0, 0}, Pi/4} { Line{46};}
Extrude {{1, 0, 0}, {0, 0, 0}, Pi/4} { Line{49};}

Point(40) = {-1,-1,-1};
Point(41) = {1, -1, -1};
Point(42) = {1, 1, -1};
Point(43) = {-1,1,-1};
Point(44) = {-1,-1,1};
Point(45) = {1, -1, 1};
Point(46) = {1, 1, 1};
Point(47) = {-1,1,1};

Line(55) = {45, 44};
Line(56) = {44, 47};
Line(57) = {47, 43};
Line(58) = {43, 40};
Line(59) = {40, 44};
Line(60) = {45, 46};
Line(61) = {46, 47};
Line(62) = {46, 42};
Line(63) = {41, 42};
Line(64) = {41, 45};
Line(65) = {41, 40};
Line(66) = {42, 43};
Line Loop(67) = {59, -55, -64, 65};
Plane Surface(68) = {67};
Line Loop(69) = {58, 59, 56, 57};
Plane Surface(70) = {69};
Line Loop(71) = {61, -56, -55, 60};
Plane Surface(72) = {71};
Line Loop(73) = {62, 66, -57, -61};
Plane Surface(74) = {73};
Line Loop(75) = {63, -62, -60, -64};
Plane Surface(76) = {75};
Line Loop(77) = {63, 66, 58, -65};
Plane Surface(78) = {77};
Surface Loop(79) = {48, 51, 54, 33, 36, 39, 42, 45, 10, 7, 4, 25, 22, 19, 16, 13};
Surface Loop(80) = {78, 76, 74, 70, 68, 72};
Volume(81) = {79, 80};
Physical Surface(82) = {48, 13, 51, 16, 19, 33, 54, 22, 25, 39, 36, 4, 7, 42, 45, 10};
Physical Surface(83) = {76, 68, 70, 72, 74, 78};
Physical Volume(84) = {81};

Mesh.CharacteristicLengthMax=.5;
