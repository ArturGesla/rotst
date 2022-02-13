#include <iostream>
#include <array>
#include <fstream>

#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace Eigen;

void rs2D();
void writeVTU_vp(double hx, double Nx, double hz, double Nz, VectorXd u);