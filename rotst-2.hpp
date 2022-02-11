#include <iostream>
#include <array>

#include <Eigen/Dense>

using namespace Eigen;

std::pair<VectorXd,double> newtonIteration(VectorXd u, double lam, double h, std::array<double,6> bc);

void save(VectorXd u, double h);
void writeVTU(double hx, double Nx, double hy, double Ny, VectorXd phi);
void rs1D();
void diff2D();
