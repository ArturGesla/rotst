#include "rs2Dfv.hpp"
#include "lean_vtk.hpp"

#include <Eigen/Eigenvalues>

#include <chrono>

using namespace std::chrono;

namespace rs2Dfv
{
    void newtonIterationRS2D(size_t Neq, size_t Nx, size_t Nz, double hx, double hz, double lam, double eps, VectorXd &u);

    void rs2D(double re, size_t nx, size_t nz, size_t nnewt, size_t nconti, double Lx, double Lz)
    {
        //size_t nx = 30; // with bc without ghost
        //size_t nz = 30; // with bc without ghost

        //fail lu if nx!=nz bug!!
        // n=4 fail lu

        size_t Neq = 4;

        size_t Nx = nx + 4; //2 ghosts
        size_t Nz = nz + 4; //2 ghosts

        //double Lx = 10;
        //double Lz = 1;

        double hx = Lx / double(nx);
        double hz = Lz / double(nz);

        double lam = 1;
        lam = re;

        double eps = hx; //from the axis - has to be fixed

        VectorXd u = VectorXd::Zero(Nx * Nz * Neq);

        for (size_t ilam = 0; ilam < nconti; ilam++)
        {
            std::cout << "============ Lam = " << lam + lam * ilam << " ============" << std::endl;
            for (size_t i = 0; i < nnewt; i++)
            {
                std::cout << "iter: " << i << std::endl;
                rs2Dfv::newtonIterationRS2D(Neq, Nx, Nz, hx, hz, lam + lam * ilam, eps, u);
            }
        }

        // for (size_t i = 0; i < 10; i++)
        // {
        //     newtonIterationRS2D(Neq, Nx, Nz, hx, hz, lam, eps, u);
        // }

        rs2Dfv::writeVTU_vp_wg(hx, Nx, hz, Nz, u);
    }

    void newtonIterationRS2D(size_t Neq, size_t Nx, size_t Nz, double hx, double hz, double lam, double eps, VectorXd &u)
    {
        // there is a problem with r=0, 1/r

        // size_t nx = 30; // with bc without ghost
        // size_t nz = 30; // with bc without ghost

        // size_t Neq = 4;

        // size_t Nx = nx + 2;
        // size_t Nz = nz + 2;

        // double Lx = 10;
        // double Lz = 1;

        // double hx = Lx / double(nx - 1);
        // double hz = Lz / double(nz - 1);

        // double lam = 1;

        // double eps = hx; //from the axis - has to be fixed

        // VectorXd u = VectorXd::Ones(Nx * Nz * Neq);

        //std::cout<<"lam"<<lam<<std::endl;

        VectorXd rhs = VectorXd::Zero(Nx * Nz * Neq);
        SparseMatrix<double> jac(Nx * Nz * Neq, Nx * Nz * Neq);

        // VectorXd f = VectorXd::Ones(Nx * Nz);
        // VectorXd g = VectorXd::Ones(Nx * Nz);
        // VectorXd h = VectorXd::Ones(Nx * Nz);
        // VectorXd p = VectorXd::Ones(Nx * Nz);
        std::vector<Triplet<double>> tripletList;

        size_t tripletSize = (Nz - 4) * (Nx - 4) * (9 + 7 + 8 + 5) + 16 + 8 * (Nz - 4) * 2 + 8 * (Nx - 4) + 8 * (Nx - 5) + 7 + 16 + (Nz - 2) * 8 + (Nx - 2) * 8;
        tripletList.reserve(tripletSize); // to do

        auto start2 = high_resolution_clock::now();

        //std::cout << "expected size: " << tripletSize << std::endl;
        //  test
        //   std::vector<double> v1 = {1.0, 2.0, 3.0};
        //   Eigen::Vector3d v2(v1.data());
        //   std::cout<<v2<<std::endl;

        {
            // x momentum
            size_t ieq = 0;
            for (size_t ix = 2; ix < Nx - 2; ix++)
            {
                for (size_t iz = 2; iz < Nz - 2; iz++)
                {
                    double r = (ix - 2) * hx + hx / 2;

                    size_t ii = (iz + ix * Nz) * Neq + ieq;

                    size_t iif = ii - ieq;
                    size_t iifxp = iif + Nz * Neq;
                    size_t iifxm = iif - Nz * Neq;
                    size_t iifzp = iif + Neq;
                    size_t iifzm = iif - Neq;

                    size_t iig = iif + 1;
                    size_t iigxp = iig + Nz * Neq;
                    size_t iigxm = iig - Nz * Neq;
                    size_t iigzp = iig + Neq;
                    size_t iigzm = iig - Neq;

                    size_t iih = iig + 1;
                    size_t iihxp = iih + Nz * Neq;
                    size_t iihxm = iih - Nz * Neq;
                    size_t iihzp = iih + Neq;
                    size_t iihzm = iih - Neq;

                    size_t iip = iih + 1;
                    size_t iipxp = iip + Nz * Neq;
                    size_t iipxm = iip - Nz * Neq;
                    size_t iipzp = iip + Neq;
                    size_t iipzm = iip - Neq;

                    //non linear
                    double first = u(iif) * (u(iifxp) - u(iifxm)) / 2.0 / hx;
                    double second = -u(iig) * u(iig) / r;
                    double third = u(iih) * (u(iifzp) - u(iifzm)) / 2.0 / hz;

                    // //linear
                    // double first = 0;
                    // double second = 0;
                    // double third = 0;

                    //linear
                    double fourth = (u(iipxp) - u(iipxm)) / 2.0 / hx;
                    //double fourth = (2 / 3.0 * u(iipxp) - 2 / 3.0 * u(iipxm)) / hx + (-1 / 12.0 * u(iipxp + Nz * Neq) + 1 / 12.0 * u(iipxm - Nz * Neq)) / hx;
                    double fifth = -u(iif) / r / r;
                    double sixth = 1 / r * (u(iifxp) - u(iifxm)) / 2.0 / hx;
                    double seventh = (u(iifxp) - 2 * u(iif) + u(iifxm)) / hx / hx;
                    double eigth = (u(iifzp) - 2 * u(iif) + u(iifzm)) / hz / hz;

                    double nlinear = first + second + third;
                    double linear = fourth - 1 / lam * (fifth + sixth + seventh + eigth);

                    rhs(ii) = nlinear + linear;

                    // linear
                    // tripletList.push_back(Triplet<double>(ii, iipxp, 1 / 2.0 / hx));
                    // tripletList.push_back(Triplet<double>(ii, iipxm, -1 / 2.0 / hx));
                    // tripletList.push_back(Triplet<double>(ii, iif, -1 / lam * (-1 / r / r - 2 / hx / hx - 2 / hz / hz)));
                    // tripletList.push_back(Triplet<double>(ii, iifxp, -1 / lam * (1 / r / 2.0 / hx + 1 / hx / hx)));
                    // tripletList.push_back(Triplet<double>(ii, iifxm, -1 / lam * (-1 / r / 2.0 / hx + 1 / hx / hx)));
                    // tripletList.push_back(Triplet<double>(ii, iifzp, -1 / lam * (1 / hz / hz)));
                    // tripletList.push_back(Triplet<double>(ii, iifzm, -1 / lam * (1 / hz / hz)));

                    // non linear

                    //2nd order pressure
                    tripletList.push_back(Triplet<double>(ii, iipxp, 1 / 2.0 / hx));
                    tripletList.push_back(Triplet<double>(ii, iipxm, -1 / 2.0 / hx));

                    //4th order pressure
                    // tripletList.push_back(Triplet<double>(ii, iipxp, 2 / 3.0 / hx));
                    // tripletList.push_back(Triplet<double>(ii, iipxm, -2 / 3.0 / hx));
                    // tripletList.push_back(Triplet<double>(ii, iipxp + Nz * Neq, -1 / 12.0 / hx));
                    // tripletList.push_back(Triplet<double>(ii, iipxm - Nz * Neq, 1 / 12.0 / hx));

                    tripletList.push_back(Triplet<double>(ii, iif, -1 / lam * (-1 / r / r - 2 / hx / hx - 2 / hz / hz) + (u(iifxp) - u(iifxm)) / 2.0 / hx));
                    tripletList.push_back(Triplet<double>(ii, iifxp, -1 / lam * (1 / r / 2.0 / hx + 1 / hx / hx) + u(iif) / 2.0 / hx));
                    tripletList.push_back(Triplet<double>(ii, iifxm, -1 / lam * (-1 / r / 2.0 / hx + 1 / hx / hx) - u(iif) / 2.0 / hx));
                    tripletList.push_back(Triplet<double>(ii, iifzp, -1 / lam * (1 / hz / hz) + u(iih) / 2.0 / hz));
                    tripletList.push_back(Triplet<double>(ii, iifzm, -1 / lam * (1 / hz / hz) - u(iih) / 2.0 / hz));

                    tripletList.push_back(Triplet<double>(ii, iig, -2 * u(iig) / r));
                    tripletList.push_back(Triplet<double>(ii, iih, (u(iifzp) - u(iifzm)) / 2.0 / hz));
                }
            }
        }

        {
            // y momentum
            size_t ieq = 1;
            for (size_t ix = 2; ix < Nx - 2; ix++)
            {
                for (size_t iz = 2; iz < Nz - 2; iz++)
                {
                    double r = (ix - 2) * hx + hx / 2;

                    size_t ii = (iz + ix * Nz) * Neq + ieq;

                    size_t iif = ii - ieq;
                    size_t iifxp = iif + Nz * Neq;
                    size_t iifxm = iif - Nz * Neq;
                    size_t iifzp = iif + Neq;
                    size_t iifzm = iif - Neq;

                    size_t iig = iif + 1;
                    size_t iigxp = iig + Nz * Neq;
                    size_t iigxm = iig - Nz * Neq;
                    size_t iigzp = iig + Neq;
                    size_t iigzm = iig - Neq;

                    size_t iih = iig + 1;
                    size_t iihxp = iih + Nz * Neq;
                    size_t iihxm = iih - Nz * Neq;
                    size_t iihzp = iih + Neq;
                    size_t iihzm = iih - Neq;

                    size_t iip = iih + 1;
                    size_t iipxp = iip + Nz * Neq;
                    size_t iipxm = iip - Nz * Neq;
                    size_t iipzp = iip + Neq;
                    size_t iipzm = iip - Neq;

                    //non linear
                    double first = u(iif) * (u(iigxp) - u(iigxm)) / 2.0 / hx;
                    double second = u(iig) * u(iif) / r;
                    double third = u(iih) * (u(iigzp) - u(iigzm)) / 2.0 / hz;

                    //linear
                    // double first = 0;
                    // double second = 0;
                    // double third = 0;

                    //linear
                    double fourth = 0;
                    double fifth = -u(iig) / r / r;
                    double sixth = 1 / r * (u(iigxp) - u(iigxm)) / 2.0 / hx;
                    double seventh = (u(iigxp) - 2 * u(iig) + u(iigxm)) / hx / hx;
                    double eigth = (u(iigzp) - 2 * u(iig) + u(iigzm)) / hz / hz;

                    double nlinear = first + second + third;
                    double linear = fourth - 1 / lam * (fifth + sixth + seventh + eigth);

                    rhs(ii) = linear + nlinear;
                    // linear
                    // tripletList.push_back(Triplet<double>(ii, iig, -1 / lam * (-1 / r / r - 2 / hx / hx - 2 / hz / hz)));
                    // tripletList.push_back(Triplet<double>(ii, iigxp, -1 / lam * (1 / r / 2.0 / hx + 1 / hx / hx)));
                    // tripletList.push_back(Triplet<double>(ii, iigxm, -1 / lam * (-1 / r / 2.0 / hx + 1 / hx / hx)));
                    // tripletList.push_back(Triplet<double>(ii, iigzp, -1 / lam * (1 / hz / hz)));
                    // tripletList.push_back(Triplet<double>(ii, iigzm, -1 / lam * (1 / hz / hz)));

                    // non linear
                    tripletList.push_back(Triplet<double>(ii, iig, -1 / lam * (-1 / r / r - 2 / hx / hx - 2 / hz / hz) + u(iif) / r));
                    tripletList.push_back(Triplet<double>(ii, iigxp, -1 / lam * (1 / r / 2.0 / hx + 1 / hx / hx) + u(iif) / 2.0 / hx));
                    tripletList.push_back(Triplet<double>(ii, iigxm, -1 / lam * (-1 / r / 2.0 / hx + 1 / hx / hx) - u(iif) / 2.0 / hx));
                    tripletList.push_back(Triplet<double>(ii, iigzp, -1 / lam * (1 / hz / hz) + u(iih) / 2.0 / hz));
                    tripletList.push_back(Triplet<double>(ii, iigzm, -1 / lam * (1 / hz / hz) - u(iih) / 2.0 / hz));

                    tripletList.push_back(Triplet<double>(ii, iif, (u(iigxp) - u(iigxm)) / 2.0 / hx + u(iig) / 2));
                    tripletList.push_back(Triplet<double>(ii, iih, (u(iigzp) - u(iigzm)) / 2.0 / hz));
                }
            }
        }

        {
            // z momentum
            size_t ieq = 2;
            for (size_t ix = 2; ix < Nx - 2; ix++)
            {
                for (size_t iz = 2; iz < Nz - 2; iz++)
                {
                    double r = (ix - 2) * hx + hx / 2;

                    size_t ii = (iz + ix * Nz) * Neq + ieq;

                    size_t iif = ii - ieq;
                    size_t iifxp = iif + Nz * Neq;
                    size_t iifxm = iif - Nz * Neq;
                    size_t iifzp = iif + Neq;
                    size_t iifzm = iif - Neq;

                    size_t iig = iif + 1;
                    size_t iigxp = iig + Nz * Neq;
                    size_t iigxm = iig - Nz * Neq;
                    size_t iigzp = iig + Neq;
                    size_t iigzm = iig - Neq;

                    size_t iih = iig + 1;
                    size_t iihxp = iih + Nz * Neq;
                    size_t iihxm = iih - Nz * Neq;
                    size_t iihzp = iih + Neq;
                    size_t iihzm = iih - Neq;

                    size_t iip = iih + 1;
                    size_t iipxp = iip + Nz * Neq;
                    size_t iipxm = iip - Nz * Neq;
                    size_t iipzp = iip + Neq;
                    size_t iipzm = iip - Neq;

                    //non linear
                    double first = u(iif) * (u(iihxp) - u(iihxm)) / 2.0 / hx;
                    double second = 0;
                    double third = u(iih) * (u(iihzp) - u(iihzm)) / 2.0 / hz;

                    //linear
                    // double first = 0;
                    // double second = 0;
                    // double third = 0;

                    //linear
                    double fourth = (u(iipzp) - u(iipzm)) / 2.0 / hz; //2nd order pressure
                    //double fourth = (2 / 3.0 * u(iipzp) - 2 / 3.0 * u(iipzm)) / hz + (-1 / 12.0 * u(iipzp + Neq) + 1 / 12.0 * u(iipzm - Neq)) / hz; //4th order pressure

                    double fifth = 0;
                    double sixth = 1 / r * (u(iihxp) - u(iihxm)) / 2.0 / hx;
                    double seventh = (u(iihxp) - 2 * u(iih) + u(iihxm)) / hx / hx;
                    double eigth = (u(iihzp) - 2 * u(iih) + u(iihzm)) / hz / hz;

                    double nlinear = first + second + third;
                    double linear = fourth - 1 / lam * (fifth + sixth + seventh + eigth);

                    rhs(ii) = linear + nlinear;

                    // //linear
                    // //2nd order pressure
                    // tripletList.push_back(Triplet<double>(ii, iipzp, 1 / 2.0 / hz));
                    // tripletList.push_back(Triplet<double>(ii, iipzm, -1 / 2.0 / hz));

                    // //4th order pressure
                    // // tripletList.push_back(Triplet<double>(ii, iipzp, 2 / 3.0 / hz));
                    // // tripletList.push_back(Triplet<double>(ii, iipzm, -2 / 3.0 / hz));
                    // // tripletList.push_back(Triplet<double>(ii, iipzp + Neq, -1 / 12.0 / hz));
                    // // tripletList.push_back(Triplet<double>(ii, iipzm - Neq, 1 / 12.0 / hz));

                    // tripletList.push_back(Triplet<double>(ii, iih, -1 / lam * (-2 / hx / hx - 2 / hz / hz)));
                    // tripletList.push_back(Triplet<double>(ii, iihxp, -1 / lam * (1 / r / 2.0 / hx + 1 / hx / hx)));
                    // tripletList.push_back(Triplet<double>(ii, iihxm, -1 / lam * (-1 / r / 2.0 / hx + 1 / hx / hx)));
                    // tripletList.push_back(Triplet<double>(ii, iihzp, -1 / lam * (1 / hz / hz)));
                    // tripletList.push_back(Triplet<double>(ii, iihzm, -1 / lam * (1 / hz / hz)));

                    //non linear
                    //2nd order pressure
                    tripletList.push_back(Triplet<double>(ii, iipzp, 1 / 2.0 / hz));
                    tripletList.push_back(Triplet<double>(ii, iipzm, -1 / 2.0 / hz));

                    //4th order pressure
                    // tripletList.push_back(Triplet<double>(ii, iipzp, 2 / 3.0 / hz));
                    // tripletList.push_back(Triplet<double>(ii, iipzm, -2 / 3.0 / hz));
                    // tripletList.push_back(Triplet<double>(ii, iipzp + Neq, -1 / 12.0 / hz));
                    // tripletList.push_back(Triplet<double>(ii, iipzm - Neq, 1 / 12.0 / hz));

                    tripletList.push_back(Triplet<double>(ii, iih, -1 / lam * (-2 / hx / hx - 2 / hz / hz) + (u(iihzp) - u(iihzm)) / 2.0 / hz));
                    tripletList.push_back(Triplet<double>(ii, iihxp, -1 / lam * (1 / r / 2.0 / hx + 1 / hx / hx) + u(iif) / 2.0 / hx));
                    tripletList.push_back(Triplet<double>(ii, iihxm, -1 / lam * (-1 / r / 2.0 / hx + 1 / hx / hx) - u(iif) / 2.0 / hx));
                    tripletList.push_back(Triplet<double>(ii, iihzp, -1 / lam * (1 / hz / hz) + u(iih) / 2.0 / hz));
                    tripletList.push_back(Triplet<double>(ii, iihzm, -1 / lam * (1 / hz / hz) - u(iih) / 2.0 / hz));

                    tripletList.push_back(Triplet<double>(ii, iif, (u(iihxp) - u(iihxm)) / 2.0 / hx));
                }
            }
        }

        {
            // conti
            size_t ieq = 3;
            for (size_t ix = 2; ix < Nx - 2; ix++)
            {
                for (size_t iz = 2; iz < Nz - 2; iz++)
                {
                    double r = (ix - 2) * hx + hx / 2;

                    size_t ii = (iz + ix * Nz) * Neq + ieq;

                    size_t iif = ii - ieq;
                    size_t iifxp = iif + Nz * Neq;
                    size_t iifxm = iif - Nz * Neq;
                    size_t iifzp = iif + Neq;
                    size_t iifzm = iif - Neq;

                    size_t iig = iif + 1;
                    size_t iigxp = iig + Nz * Neq;
                    size_t iigxm = iig - Nz * Neq;
                    size_t iigzp = iig + Neq;
                    size_t iigzm = iig - Neq;

                    size_t iih = iig + 1;
                    size_t iihxp = iih + Nz * Neq;
                    size_t iihxm = iih - Nz * Neq;
                    size_t iihzp = iih + Neq;
                    size_t iihzm = iih - Neq;

                    size_t iip = iih + 1;
                    size_t iipxp = iip + Nz * Neq;
                    size_t iipxm = iip - Nz * Neq;
                    size_t iipzp = iip + Neq;
                    size_t iipzm = iip - Neq;

                    double first = 1 / r * u(iif);
                    double second = (u(iifxp) - u(iifxm)) / 2.0 / hx;
                    double third = (u(iihzp) - u(iihzm)) / 2.0 / hz;

                    double linear = first + second + third;

                    rhs(ii) = linear;

                    tripletList.push_back(Triplet<double>(ii, iif, 1 / r));
                    tripletList.push_back(Triplet<double>(ii, iifxp, 1 / 2.0 / hx));
                    tripletList.push_back(Triplet<double>(ii, iifxm, -1 / 2.0 / hx));
                    tripletList.push_back(Triplet<double>(ii, iihzp, 1 / 2.0 / hz));
                    tripletList.push_back(Triplet<double>(ii, iihzm, -1 / 2.0 / hz));
                }
            }
        }

        // Boundaries
        //  Dirichlet BC on walls
        double fLeft = 0;
        double fRight = 0;
        double fTop = 0;
        double fBottom = 0;

        double gLeft = 0;
        double gRight = 0;
        double gTop = 1;
        double gBottom = 0;

        double hLeft = 0;
        double hRight = 0;
        double hTop = 0;
        double hBottom = 0;

        // double dpLeft = 0;
        // double dpRight = 0;
        // double dpTop = 0;
        // double dpBottom = 0;

        double pRef = 1;

        double cval = 0;

        {
            // 4 corner points are constant for all the BC
            for (size_t ieq = 0; ieq < Neq; ieq++)
            {
                //left top corner
                size_t ix = 1;
                size_t iz = 1;

                size_t ii = (iz + ix * Nz) * Neq + ieq;

                rhs(ii) = u(ii) - cval;

                tripletList.push_back(Triplet<double>(ii, ii, 1));
            }
            for (size_t ieq = 0; ieq < Neq; ieq++)
            {
                //left bot corner
                size_t ix = 1;
                size_t iz = Nz - 2;

                size_t ii = (iz + ix * Nz) * Neq + ieq;

                rhs(ii) = u(ii) - cval;

                tripletList.push_back(Triplet<double>(ii, ii, 1));
            }
            for (size_t ieq = 0; ieq < Neq; ieq++)
            {
                //right top corner
                size_t ix = Nx - 2;
                size_t iz = 1;

                size_t ii = (iz + ix * Nz) * Neq + ieq;

                rhs(ii) = u(ii) - cval;

                tripletList.push_back(Triplet<double>(ii, ii, 1));
            }
            for (size_t ieq = 0; ieq < Neq; ieq++)
            {
                //right bot corner
                size_t ix = Nx - 2;
                size_t iz = Nz - 2;

                size_t ii = (iz + ix * Nz) * Neq + ieq;

                rhs(ii) = u(ii) - cval;

                tripletList.push_back(Triplet<double>(ii, ii, 1));
            }
        }

        {
            //General BC at the walls - not in the corners

            // left
            for (size_t ix = 1; ix < 2; ix++)
            {
                for (size_t iz = 2; iz < Nz - 2; iz++)
                {
                    size_t ii = (iz + ix * Nz) * Neq;

                    size_t iif = ii;
                    size_t iifxp = iif + Nz * Neq;
                    size_t iifxm = iif - Nz * Neq;
                    size_t iifzp = iif + Neq;
                    size_t iifzm = iif - Neq;

                    size_t iig = iif + 1;
                    size_t iigxp = iig + Nz * Neq;
                    size_t iigxm = iig - Nz * Neq;
                    size_t iigzp = iig + Neq;
                    size_t iigzm = iig - Neq;

                    size_t iih = iig + 1;
                    size_t iihxp = iih + Nz * Neq;
                    size_t iihxm = iih - Nz * Neq;
                    size_t iihzp = iih + Neq;
                    size_t iihzm = iih - Neq;

                    size_t iip = iih + 1;
                    size_t iipxp = iip + Nz * Neq;
                    size_t iipxm = iip - Nz * Neq;
                    size_t iipzp = iip + Neq;
                    size_t iipzm = iip - Neq;

                    rhs(iif) = (u(iif) + u(iifxp)) / 2.0 - fLeft;
                    rhs(iig) = (u(iig) + u(iigxp)) / 2.0 - gLeft;
                    rhs(iih) = (u(iih) + u(iihxp)) / 2.0 - hLeft;
                    rhs(iip) = (-u(iip) + u(iipxp)) / hx;

                    tripletList.push_back(Triplet<double>(iif, iif, 1 / 2.0));
                    tripletList.push_back(Triplet<double>(iif, iifxp, 1 / 2.0));
                    tripletList.push_back(Triplet<double>(iig, iig, 1 / 2.0));
                    tripletList.push_back(Triplet<double>(iig, iigxp, 1 / 2.0));
                    tripletList.push_back(Triplet<double>(iih, iih, 1 / 2.0));
                    tripletList.push_back(Triplet<double>(iih, iihxp, 1 / 2.0));

                    tripletList.push_back(Triplet<double>(iip, iip, -1 / hx));
                    tripletList.push_back(Triplet<double>(iip, iipxp, 1 / hx));
                }
            }

            // right
            for (size_t ix = Nx - 2; ix < Nx - 1; ix++)
            {
                for (size_t iz = 2; iz < Nz - 2; iz++)
                {
                    size_t ii = (iz + ix * Nz) * Neq;

                    size_t iif = ii;
                    size_t iifxp = iif + Nz * Neq;
                    size_t iifxm = iif - Nz * Neq;
                    size_t iifzp = iif + Neq;
                    size_t iifzm = iif - Neq;

                    size_t iig = iif + 1;
                    size_t iigxp = iig + Nz * Neq;
                    size_t iigxm = iig - Nz * Neq;
                    size_t iigzp = iig + Neq;
                    size_t iigzm = iig - Neq;

                    size_t iih = iig + 1;
                    size_t iihxp = iih + Nz * Neq;
                    size_t iihxm = iih - Nz * Neq;
                    size_t iihzp = iih + Neq;
                    size_t iihzm = iih - Neq;

                    size_t iip = iih + 1;
                    size_t iipxp = iip + Nz * Neq;
                    size_t iipxm = iip - Nz * Neq;
                    size_t iipzp = iip + Neq;
                    size_t iipzm = iip - Neq;

                    rhs(iif) = (u(iif) + u(iifxm)) / 2.0 - fRight;
                    rhs(iig) = (u(iig) + u(iigxm)) / 2.0 - gRight;
                    rhs(iih) = (u(iih) + u(iihxm)) / 2.0 - hRight;
                    rhs(iip) = (u(iip) - u(iipxm)) / hx;

                    tripletList.push_back(Triplet<double>(iif, iif, 1 / 2.0));
                    tripletList.push_back(Triplet<double>(iif, iifxm, 1 / 2.0));
                    tripletList.push_back(Triplet<double>(iig, iig, 1 / 2.0));
                    tripletList.push_back(Triplet<double>(iig, iigxm, 1 / 2.0));
                    tripletList.push_back(Triplet<double>(iih, iih, 1 / 2.0));
                    tripletList.push_back(Triplet<double>(iig, iihxm, 1 / 2.0));

                    tripletList.push_back(Triplet<double>(iip, iip, 1 / hx));
                    tripletList.push_back(Triplet<double>(iip, iipxm, -1 / hx));
                }
            }

            // top
            for (size_t ix = 2; ix < Nx - 2; ix++)
            {
                for (size_t iz = 1; iz < 2; iz++)
                {
                    double r = (ix - 2) * hx + hx / 2;

                    size_t ii = (iz + ix * Nz) * Neq;

                    size_t iif = ii;
                    size_t iifxp = iif + Nz * Neq;
                    size_t iifxm = iif - Nz * Neq;
                    size_t iifzp = iif + Neq;
                    size_t iifzm = iif - Neq;

                    size_t iig = iif + 1;
                    size_t iigxp = iig + Nz * Neq;
                    size_t iigxm = iig - Nz * Neq;
                    size_t iigzp = iig + Neq;
                    size_t iigzm = iig - Neq;

                    size_t iih = iig + 1;
                    size_t iihxp = iih + Nz * Neq;
                    size_t iihxm = iih - Nz * Neq;
                    size_t iihzp = iih + Neq;
                    size_t iihzm = iih - Neq;

                    size_t iip = iih + 1;
                    size_t iipxp = iip + Nz * Neq;
                    size_t iipxm = iip - Nz * Neq;
                    size_t iipzp = iip + Neq;
                    size_t iipzm = iip - Neq;

                    rhs(iif) = (u(iif) + u(iifzp)) / 2.0 - fTop;
                    rhs(iig) = (u(iig) + u(iigzp)) / 2.0 - gTop * r;
                    rhs(iih) = (u(iih) + u(iihzp)) / 2.0 - hTop;
                    rhs(iip) = (-u(iip) + u(iipzp)) / hz;

                    tripletList.push_back(Triplet<double>(iif, iif, 1 / 2.0));
                    tripletList.push_back(Triplet<double>(iif, iifzp, 1 / 2.0));
                    tripletList.push_back(Triplet<double>(iig, iig, 1 / 2.0));
                    tripletList.push_back(Triplet<double>(iig, iigzp, 1 / 2.0));
                    tripletList.push_back(Triplet<double>(iih, iih, 1 / 2.0));
                    tripletList.push_back(Triplet<double>(iih, iihzp, 1 / 2.0));

                    tripletList.push_back(Triplet<double>(iip, iip, -1 / hz));
                    tripletList.push_back(Triplet<double>(iip, iipzp, 1 / hz));
                }
            }
            // bottom without last point - thiw will be p pinning
            for (size_t ix = 2; ix < Nx - 3; ix++)
            {
                for (size_t iz = Nz - 2; iz < Nz - 1; iz++)
                {
                    size_t ii = (iz + ix * Nz) * Neq;

                    size_t iif = ii;
                    size_t iifxp = iif + Nz * Neq;
                    size_t iifxm = iif - Nz * Neq;
                    size_t iifzp = iif + Neq;
                    size_t iifzm = iif - Neq;

                    size_t iig = iif + 1;
                    size_t iigxp = iig + Nz * Neq;
                    size_t iigxm = iig - Nz * Neq;
                    size_t iigzp = iig + Neq;
                    size_t iigzm = iig - Neq;

                    size_t iih = iig + 1;
                    size_t iihxp = iih + Nz * Neq;
                    size_t iihxm = iih - Nz * Neq;
                    size_t iihzp = iih + Neq;
                    size_t iihzm = iih - Neq;

                    size_t iip = iih + 1;
                    size_t iipxp = iip + Nz * Neq;
                    size_t iipxm = iip - Nz * Neq;
                    size_t iipzp = iip + Neq;
                    size_t iipzm = iip - Neq;

                    rhs(iif) = (u(iif) + u(iifzm)) / 2.0 - fBottom;
                    rhs(iig) = (u(iig) + u(iigzm)) / 2.0 - gBottom;
                    rhs(iih) = (u(iih) + u(iihzm)) / 2.0 - hBottom;
                    rhs(iip) = (u(iip) - u(iipzm)) / hz;

                    tripletList.push_back(Triplet<double>(iif, iif, 1 / 2.0));
                    tripletList.push_back(Triplet<double>(iif, iifzm, 1 / 2.0));
                    tripletList.push_back(Triplet<double>(iig, iig, 1 / 2.0));
                    tripletList.push_back(Triplet<double>(iig, iigzm, 1 / 2.0));
                    tripletList.push_back(Triplet<double>(iih, iih, 1 / 2.0));
                    tripletList.push_back(Triplet<double>(iih, iihzm, 1 / 2.0));

                    tripletList.push_back(Triplet<double>(iip, iip, 1 / hz));
                    tripletList.push_back(Triplet<double>(iip, iipzm, -1 / hz));
                }
            }
            // p pinning
            for (size_t ix = Nx - 3; ix < Nx - 2; ix++)
            {
                for (size_t iz = Nz - 2; iz < Nz - 1; iz++)
                {
                    size_t ii = (iz + ix * Nz) * Neq;

                    size_t iif = ii;
                    size_t iifxp = iif + Nz * Neq;
                    size_t iifxm = iif - Nz * Neq;
                    size_t iifzp = iif + Neq;
                    size_t iifzm = iif - Neq;

                    size_t iig = iif + 1;
                    size_t iigxp = iig + Nz * Neq;
                    size_t iigxm = iig - Nz * Neq;
                    size_t iigzp = iig + Neq;
                    size_t iigzm = iig - Neq;

                    size_t iih = iig + 1;
                    size_t iihxp = iih + Nz * Neq;
                    size_t iihxm = iih - Nz * Neq;
                    size_t iihzp = iih + Neq;
                    size_t iihzm = iih - Neq;

                    size_t iip = iih + 1;
                    size_t iipxp = iip + Nz * Neq;
                    size_t iipxm = iip - Nz * Neq;
                    size_t iipzp = iip + Neq;
                    size_t iipzm = iip - Neq;

                    rhs(iif) = (u(iif) + u(iifzm)) / 2.0 - fBottom;
                    rhs(iig) = (u(iig) + u(iigzm)) / 2.0 - gBottom;
                    rhs(iih) = (u(iih) + u(iihzm)) / 2.0 - hBottom;
                    rhs(iip) = u(iip) - pRef;

                    tripletList.push_back(Triplet<double>(iif, iif, 1 / 2.0));
                    tripletList.push_back(Triplet<double>(iif, iifzm, 1 / 2.0));
                    tripletList.push_back(Triplet<double>(iig, iig, 1 / 2.0));
                    tripletList.push_back(Triplet<double>(iig, iigzm, 1 / 2.0));
                    tripletList.push_back(Triplet<double>(iih, iih, 1 / 2.0));
                    tripletList.push_back(Triplet<double>(iig, iihzm, 1 / 2.0));

                    tripletList.push_back(Triplet<double>(iip, iip, 1));
                }
            }
        }

        {
            // 4 corner points are constant for all the BC - OUTER GHOST
            for (size_t ieq = 0; ieq < Neq; ieq++)
            {
                //left top corner
                size_t ix = 0;
                size_t iz = 0;

                size_t ii = (iz + ix * Nz) * Neq + ieq;

                rhs(ii) = u(ii) - cval;

                tripletList.push_back(Triplet<double>(ii, ii, 1));
            }
            for (size_t ieq = 0; ieq < Neq; ieq++)
            {
                //left bot corner
                size_t ix = 0;
                size_t iz = Nz - 1;

                size_t ii = (iz + ix * Nz) * Neq + ieq;

                rhs(ii) = u(ii) - cval;

                tripletList.push_back(Triplet<double>(ii, ii, 1));
            }
            for (size_t ieq = 0; ieq < Neq; ieq++)
            {
                //right top corner
                size_t ix = Nx - 1;
                size_t iz = 0;

                size_t ii = (iz + ix * Nz) * Neq + ieq;

                rhs(ii) = u(ii) - cval;

                tripletList.push_back(Triplet<double>(ii, ii, 1));
            }
            for (size_t ieq = 0; ieq < Neq; ieq++)
            {
                //right bot corner
                size_t ix = Nx - 1;
                size_t iz = Nz - 1;

                size_t ii = (iz + ix * Nz) * Neq + ieq;

                rhs(ii) = u(ii) - cval;

                tripletList.push_back(Triplet<double>(ii, ii, 1));
            }
        }

        //outer ghost layer
        // top
        for (size_t ix = 1; ix < Nx - 1; ix++)
        {
            for (size_t iz = 0; iz < 1; iz++)
            {
                size_t ii = (iz + ix * Nz) * Neq;

                size_t iif = ii;
                size_t iifxp = iif + Nz * Neq;
                size_t iifxm = iif - Nz * Neq;
                size_t iifzp = iif + Neq;
                size_t iifzm = iif - Neq;

                size_t iig = iif + 1;
                size_t iigxp = iig + Nz * Neq;
                size_t iigxm = iig - Nz * Neq;
                size_t iigzp = iig + Neq;
                size_t iigzm = iig - Neq;

                size_t iih = iig + 1;
                size_t iihxp = iih + Nz * Neq;
                size_t iihxm = iih - Nz * Neq;
                size_t iihzp = iih + Neq;
                size_t iihzm = iih - Neq;

                size_t iip = iih + 1;
                size_t iipxp = iip + Nz * Neq;
                size_t iipxm = iip - Nz * Neq;
                size_t iipzp = iip + Neq;
                size_t iipzm = iip - Neq;

                rhs(iif) = u(iif) - cval;
                rhs(iig) = u(iig) - cval;
                rhs(iih) = u(iih) - cval;

                rhs(iip) = u(iip) - cval;

                tripletList.push_back(Triplet<double>(iif, iif, 1));
                tripletList.push_back(Triplet<double>(iig, iig, 1));
                tripletList.push_back(Triplet<double>(iih, iih, 1));

                tripletList.push_back(Triplet<double>(iip, iip, 1));
            }
        }

        // bot
        for (size_t ix = 1; ix < Nx - 1; ix++)
        {
            for (size_t iz = Nz - 1; iz < Nz; iz++)
            {
                size_t ii = (iz + ix * Nz) * Neq;

                size_t iif = ii;
                size_t iifxp = iif + Nz * Neq;
                size_t iifxm = iif - Nz * Neq;
                size_t iifzp = iif + Neq;
                size_t iifzm = iif - Neq;

                size_t iig = iif + 1;
                size_t iigxp = iig + Nz * Neq;
                size_t iigxm = iig - Nz * Neq;
                size_t iigzp = iig + Neq;
                size_t iigzm = iig - Neq;

                size_t iih = iig + 1;
                size_t iihxp = iih + Nz * Neq;
                size_t iihxm = iih - Nz * Neq;
                size_t iihzp = iih + Neq;
                size_t iihzm = iih - Neq;

                size_t iip = iih + 1;
                size_t iipxp = iip + Nz * Neq;
                size_t iipxm = iip - Nz * Neq;
                size_t iipzp = iip + Neq;
                size_t iipzm = iip - Neq;

                rhs(iif) = u(iif) - cval;
                rhs(iig) = u(iig) - cval;
                rhs(iih) = u(iih) - cval;

                rhs(iip) = u(iip) - cval;

                tripletList.push_back(Triplet<double>(iif, iif, 1));
                tripletList.push_back(Triplet<double>(iig, iig, 1));
                tripletList.push_back(Triplet<double>(iih, iih, 1));

                tripletList.push_back(Triplet<double>(iip, iip, 1));
            }
        }

        //left
        for (size_t ix = 0; ix < 1; ix++)
        {
            for (size_t iz = 1; iz < Nz - 1; iz++)
            {
                size_t ii = (iz + ix * Nz) * Neq;

                size_t iif = ii;
                size_t iifxp = iif + Nz * Neq;
                size_t iifxm = iif - Nz * Neq;
                size_t iifzp = iif + Neq;
                size_t iifzm = iif - Neq;

                size_t iig = iif + 1;
                size_t iigxp = iig + Nz * Neq;
                size_t iigxm = iig - Nz * Neq;
                size_t iigzp = iig + Neq;
                size_t iigzm = iig - Neq;

                size_t iih = iig + 1;
                size_t iihxp = iih + Nz * Neq;
                size_t iihxm = iih - Nz * Neq;
                size_t iihzp = iih + Neq;
                size_t iihzm = iih - Neq;

                size_t iip = iih + 1;
                size_t iipxp = iip + Nz * Neq;
                size_t iipxm = iip - Nz * Neq;
                size_t iipzp = iip + Neq;
                size_t iipzm = iip - Neq;

                rhs(iif) = u(iif) - cval;
                rhs(iig) = u(iig) - cval;
                rhs(iih) = u(iih) - cval;

                rhs(iip) = u(iip) - cval;

                tripletList.push_back(Triplet<double>(iif, iif, 1));
                tripletList.push_back(Triplet<double>(iig, iig, 1));
                tripletList.push_back(Triplet<double>(iih, iih, 1));

                tripletList.push_back(Triplet<double>(iip, iip, 1));
            }
        }

        //right
        for (size_t ix = Nx - 1; ix < Nx; ix++)
        {
            for (size_t iz = 1; iz < Nz - 1; iz++)
            {
                size_t ii = (iz + ix * Nz) * Neq;

                size_t iif = ii;
                size_t iifxp = iif + Nz * Neq;
                size_t iifxm = iif - Nz * Neq;
                size_t iifzp = iif + Neq;
                size_t iifzm = iif - Neq;

                size_t iig = iif + 1;
                size_t iigxp = iig + Nz * Neq;
                size_t iigxm = iig - Nz * Neq;
                size_t iigzp = iig + Neq;
                size_t iigzm = iig - Neq;

                size_t iih = iig + 1;
                size_t iihxp = iih + Nz * Neq;
                size_t iihxm = iih - Nz * Neq;
                size_t iihzp = iih + Neq;
                size_t iihzm = iih - Neq;

                size_t iip = iih + 1;
                size_t iipxp = iip + Nz * Neq;
                size_t iipxm = iip - Nz * Neq;
                size_t iipzp = iip + Neq;
                size_t iipzm = iip - Neq;

                rhs(iif) = u(iif) - cval;
                rhs(iig) = u(iig) - cval;
                rhs(iih) = u(iih) - cval;

                rhs(iip) = u(iip) - cval;

                tripletList.push_back(Triplet<double>(iif, iif, 1));
                tripletList.push_back(Triplet<double>(iig, iig, 1));
                tripletList.push_back(Triplet<double>(iih, iih, 1));

                tripletList.push_back(Triplet<double>(iip, iip, 1));
            }
        }

        // for (size_t i = 0; i < tripletList.size(); i++)
        // {
        //     std::cout << tripletList[i].row() << " " << tripletList[i].col() << " " << tripletList[i].value() << std::endl;
        // }
        // return;

        //std::cout<<a.toDenseMatrix()<<std::endl;

        jac.setFromTriplets(tripletList.begin(), tripletList.end());

        // for (size_t i = 0; i < tripletList.size(); i++)
        // {
        //     std::cout << tripletList[i].row() << " " << tripletList[i].col() << " " << tripletList[i].value() << std::endl;
        // }

        // std::cout << jac << std::endl;

        // std::cout << rhs << std::endl;
        // std::cout << jac << std::endl;
        //std::cout << "rl size: " << tripletList.size() << std::endl;
        auto stop2 = high_resolution_clock::now();
        auto duration2 = duration_cast<microseconds>(stop2 - start2);
        std::cout << "init:" << duration2.count() << std::endl;

        auto start = high_resolution_clock::now();

        SparseLU<SparseMatrix<double>> solver;
        //BiCGSTAB<SparseMatrix<double>, IncompleteLUT<double>> solver; //does not work for zero ig
        //solver.setTolerance(1E-1);

        solver.compute(jac);

        if (solver.info() != Success)
        {
            throw std::invalid_argument("LU failed.");
        }

        VectorXd du = solver.solve(-rhs);

        if (solver.info() != Success)
        {
            throw std::invalid_argument("Solve failed.");
        }

        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        std::cout << "solve:" << duration.count() << std::endl;

        u = u + du;
        {
            double res = (jac * du + rhs).norm();
            std::cout << "equations solved with res:" << res << std::endl;
        }
        {
            double res = (rhs).norm();
            std::cout << "Norm of prev rhs:" << res << std::endl;
        }
    }

    void writeVTU_vp(double hx, double Nx, double hz, double Nz, VectorXd data)
    {
        // his data
        std::vector<double> points = {
            1., 1., -1.,
            1., -1., 1.,
            -1., -1., 0.};
        std::vector<int> elements = {0, 1, 2};
        std::vector<double> scalar_field = {0., 1., 2.};
        std::vector<double> vector_field = points;

        const int dim = 3;
        const int cell_size = 4;
        std::string filename = "rs2D.vtu";
        leanvtk::VTUWriter writer;

        // my data
        std::vector<double> points2;
        std::vector<int> elements2;
        std::vector<double> u;
        std::vector<double> v;
        std::vector<double> w;
        std::vector<double> p;

        std::vector<double> vel;

        for (size_t ix = 1 + 1; ix < Nx - 1 - 1; ix++)
        {
            for (size_t iz = 1 + 1; iz < Nz - 1 - 1; iz++)
            {
                double z = 0;
                int ii = iz + Nz * ix;

                points2.push_back((ix - 2) * hx);
                points2.push_back(0);
                points2.push_back((iz - 2) * hz);

                u.push_back(data(ii * 4));
                v.push_back(data(ii * 4 + 1));
                w.push_back(data(ii * 4 + 2));
                p.push_back(data(ii * 4 + 3));

                vel.push_back(data(ii * 4));
                vel.push_back(data(ii * 4 + 1));
                vel.push_back(data(ii * 4 + 2));
            }
        }

        for (size_t ix = 1 + 1; ix < Nx - 1 - 1 - 1; ix++)
        {
            for (size_t iz = 1 + 1; iz < Nz - 1 - 1 - 1; iz++)
            {
                double z = 0;
                int ii = (iz - 2) + (Nz - 4) * (ix - 2);
                // myfile << ix * hx << ",\t\t" << (Nz - 2 - iz) * hz << ",\t\t" << z << ",\t\t" << phi(ii) << std::endl;

                elements2.push_back(ii);
                elements2.push_back(ii + 1);
                elements2.push_back(ii + 1 + Nz - 2 - 2);
                elements2.push_back(ii + 1 + Nz - 3 - 2);
            }
        }

        writer.add_scalar_field("u", u);
        writer.add_scalar_field("v", v);
        writer.add_scalar_field("w", w);
        writer.add_scalar_field("p", p);

        writer.add_vector_field("vel", vel, dim);

        // writer.add_vector_field("vector_field", vector_field, dim);

        // writer.write_point_cloud(filename, dim, points2);
        writer.write_surface_mesh(filename, dim, cell_size, points2, elements2);
    }
    void writeVTU_vp_wg(double hx, double Nx, double hz, double Nz, VectorXd data) //with ghost
    {
        // his data
        std::vector<double> points = {
            1., 1., -1.,
            1., -1., 1.,
            -1., -1., 0.};
        std::vector<int> elements = {0, 1, 2};
        std::vector<double> scalar_field = {0., 1., 2.};
        std::vector<double> vector_field = points;

        const int dim = 3;
        const int cell_size = 4;
        std::string filename = "rs2D.vtu";
        leanvtk::VTUWriter writer;

        // my data
        std::vector<double> points2;
        std::vector<int> elements2;
        std::vector<double> u;
        std::vector<double> v;
        std::vector<double> w;
        std::vector<double> p;

        std::vector<double> vel;

        for (size_t ix = 0; ix < Nx; ix++)
        {
            for (size_t iz = 0; iz < Nz; iz++)
            {
                double y = 0;
                int ii = iz + Nz * ix;
                double x = (ix)*hx - 2 * hx;
                double z = (iz)*hz - 2 * hz;

                //left top
                points2.push_back(x); //x
                points2.push_back(y); //y
                points2.push_back(z); //z

                //left bot
                points2.push_back(x);      //x
                points2.push_back(y);      //y
                points2.push_back(z + hz); //z

                //right bot
                points2.push_back(x + hx); //x
                points2.push_back(y);      //y
                points2.push_back(z + hz); //z

                //right top
                points2.push_back(x + hx); //x
                points2.push_back(y);      //y
                points2.push_back(z);      //z
            }
        }

        for (size_t ix = 0; ix < Nx; ix++)
        {
            for (size_t iz = 0; iz < Nz; iz++)
            {
                double z = 0;
                size_t ii = iz + Nz * ix;

                u.push_back(data(ii * 4));
                v.push_back(data(ii * 4 + 1));
                w.push_back(data(ii * 4 + 2));
                p.push_back(data(ii * 4 + 3));

                vel.push_back(data(ii * 4));
                vel.push_back(data(ii * 4 + 1));
                vel.push_back(data(ii * 4 + 2));

                size_t ii_pt = ii * 4;

                elements2.push_back(ii_pt);
                elements2.push_back(ii_pt + 1);
                elements2.push_back(ii_pt + 2);
                elements2.push_back(ii_pt + 3);
            }
        }

        // std::cout<<points2.size()<<"\n"<<u.size()<<"\n"<<elements2.size()<<std::endl;
        // return;

        writer.add_cell_scalar_field("u", u);
        writer.add_cell_scalar_field("v", v);
        writer.add_cell_scalar_field("w", w);
        writer.add_cell_scalar_field("p", p);

        writer.add_cell_vector_field("vel", vel, dim);

        // writer.add_vector_field("vector_field", vector_field, dim);

        // writer.write_point_cloud(filename, dim, points2);
        writer.write_surface_mesh(filename, dim, cell_size, points2, elements2);
    }

    // output if indices
    //  std::cout << "loop" << std::endl
    //                            << ix << std::endl
    //                            << ix << std::endl
    //                            << ii << std::endl
    //                            << "f" << std::endl
    //                            << iif << std::endl
    //                            << iifxp << std::endl
    //                            << iifxm << std::endl
    //                            << iifzp << std::endl
    //                            << iifym << std::endl
    //                            << "g" << std::endl
    //                            << iig << std::endl
    //                            << iigxp << std::endl
    //                            << iigxm << std::endl
    //                            << iigyp << std::endl
    //                            << iigym << std::endl
    //                            << "h" << std::endl
    //                            << iih << std::endl
    //                            << iihxp << std::endl
    //                            << iihxm << std::endl
    //                            << iihyp << std::endl
    //                            << iihym << std::endl
    //                            << "p" << std::endl
    //                            << iip << std::endl
    //                            << iipxp << std::endl
    //                            << iipxm << std::endl
    //                            << iipyp << std::endl
    //                            << iipym << std::endl;

    // // MatrixXd A = MatrixXd::Zero(Nx * Nz * Neq, Nx * Nz * Neq);
    // // for (size_t i = 0; i < tripletList.size(); i++)
    // // {

    // //     A(tripletList[i].row(), tripletList[i].col()) = tripletList[i].value();
    // // }

    // // Eigen::ComplexEigenSolver<Eigen::MatrixXd> ges;
    // // ges.compute(A);
    // // std::cout << ges.info() << std::endl;

    // // Eigen::VectorXd alpha_real = Eigen::VectorXd(ges.eigenvalues().real());
    // // Eigen::VectorXd alpha_imag = Eigen::VectorXd(ges.eigenvalues().imag());

    // // for (int i = 0; i < alpha_real.rows(); i++)
    // // {
    // //     std::cout << "Real: " << alpha_real(i) << "	Imag: " << alpha_imag(i) << std::endl;
    // // }

    // //
    // Eigen::GeneralizedEigenSolver<Eigen::MatrixXd> ges;
    // MatrixXd B = MatrixXd::Identity(Nx * Nz * Neq, Nx * Nz * Neq);
    // MatrixXd A = MatrixXd(jac);
    // std::cout<<A<<std::endl;
    // std::cout<<B<<std::endl;
    // ges.compute(A, B);
    // std::cout << ges.info() << std::endl;

    // Eigen::VectorXd alpha_real = Eigen::VectorXd(ges.alphas().real());
    // Eigen::VectorXd alpha_imag = Eigen::VectorXd(ges.alphas().imag());
    // Eigen::VectorXd beta = Eigen::VectorXd(ges.betas());

    // for (int i = 0; i < alpha_real.rows(); i++)
    // {
    //     std::cout << "Real: " << alpha_real(i) / beta(i) << "	Imag: " << alpha_imag(i) / beta(i) << std::endl;
    // }

    // FullPivLU<MatrixXd> lu_decomp(MatrixXd(jac));
    // auto rank = lu_decomp.rank();
    // std::cout << rank << std::endl;

}
