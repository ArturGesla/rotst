#include <iostream>
#include <array>
#include <fstream>

//EIGEN_USE_MKL_ALL

#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace Eigen;

namespace rs2Dfv
{
    void rs2D(double re = 1, size_t nx = 30, size_t nz = 30, size_t nnewt = 5, size_t nconti = 1, double Lx = 10, double Lz = 1);
    void writeVTU_vp(double hx, double Nx, double hz, double Nz, VectorXd u);
    void writeVTU_vp_wg(double hx, double Nx, double hz, double Nz, VectorXd data); //with ghost

}
