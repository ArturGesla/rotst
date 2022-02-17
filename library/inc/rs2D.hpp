#include <iostream>
#include <array>
#include <fstream>

#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace Eigen;

void rs2D(double re=1, size_t nx=30, size_t nz=30, size_t nnewt=5, size_t nconti=1);
void writeVTU_vp(double hx, double Nx, double hz, double Nz, VectorXd u);