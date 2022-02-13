#include <iostream>
#include <array>
#include <fstream>

#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace Eigen;

void diff2D();
void writeVTU(double hx, double Nx, double hy, double Ny, VectorXd phi);

