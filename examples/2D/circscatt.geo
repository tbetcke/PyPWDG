Point(1) = {0, 0, 0};
Point(2) = {.5, 0, 0};
Point(3) = {2, 0, 0};
Point(4) = {0, .5, 0};
Point(5) = {0, 2, 0};
Point(6) = {-.5, 0, 0};
Point(7) = {-2, 0, 0};
Point(8) = {0, -.5, 0};
Point(9) = {0, -2, 0};
Circle(1) = {2, 1, 4};
Circle(2) = {4, 1, 6};
Circle(3) = {6, 1, 8};
Circle(4) = {8, 1, 2};
Circle(5) = {3, 1, 5};
Circle(6) = {5, 1, 7};
Circle(7) = {7, 1, 9};
Circle(8) = {9, 1, 3};
Line Loop(9) = {5, 6, 7, 8};
Line Loop(10) = {2, 3, 4, 1};
Plane Surface(11) = {9, 10};
Physical Line(12) = {5, 6, 7, 8};
Physical Line(13) = {1, 2, 3, 4};
Physical Surface(14) = {11};
