Point(1) = {0, 0, 0, 1e+22};
Point(2) = {1, 0, 0, 1e+22};
Point(3) = {1, 0, 0, 1e+22};
Point(4) = {1, 1, 0, 1e+22};
Point(5) = {0, 1, 0, 1e+22};
Line(1) = {4, 2};
Line(2) = {2, 1};
Line(3) = {1, 5};
Line(4) = {5, 4};
Line Loop(6) = {1, 2, 3, 4};
Plane Surface(6) = {6};
Physical Line(7) = {1};
Physical Line(8) = {4};
Physical Line(9) = {3};
Physical Line(10) = {2};
Physical Surface(11) = {6};
Translate {1, 0, 0} {
  Point{5, 4, 2, 1};
}
