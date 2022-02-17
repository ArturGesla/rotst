#include "rs2D.hpp"
#include "lean_vtk.hpp"

#include <Eigen/Eigenvalues>

#include <chrono>

using namespace std::chrono;

void newtonIterationRS2D(size_t Neq, size_t Nx, size_t Nz, double hx, double hz, double lam, double eps, VectorXd &u);

void rs2D(double re, size_t nx, size_t nz, size_t nnewt, size_t nconti)
{
    //size_t nx = 30; // with bc without ghost
    //size_t nz = 30; // with bc without ghost

    //fail lu if nx!=nz bug!!
    // n=4 fail lu

    size_t Neq = 4;

    size_t Nx = nx + 4; //2 ghosts
    size_t Nz = nz + 4; //2 ghosts

    double Lx = 10;
    double Lz = 1;

    double hx = Lx / double(nx - 1);
    double hz = Lz / double(nz - 1);

    double lam = 1;
    lam = re;

    double eps = hx; //from the axis - has to be fixed

    VectorXd u = VectorXd::Zero(Nx * Nz * Neq);

    for (size_t ilam = 0; ilam < nconti; ilam++)
    {
        std::cout << "============ Lam = " << lam + lam * ilam << " ============"<<std::endl;
        for (size_t i = 0; i < nnewt; i++)
        {
            std::cout << "iter: " << i << std::endl;
            newtonIterationRS2D(Neq, Nx, Nz, hx, hz, lam + lam * ilam, eps, u);
        }
    }

    // for (size_t i = 0; i < 10; i++)
    // {
    //     newtonIterationRS2D(Neq, Nx, Nz, hx, hz, lam, eps, u);
    // }

    writeVTU_vp(hx, Nx, hz, Nz, u);
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

    VectorXd rhs = VectorXd::Zero(Nx * Nz * Neq);
    SparseMatrix<double> jac(Nx * Nz * Neq, Nx * Nz * Neq);

    // VectorXd f = VectorXd::Ones(Nx * Nz);
    // VectorXd g = VectorXd::Ones(Nx * Nz);
    // VectorXd h = VectorXd::Ones(Nx * Nz);
    // VectorXd p = VectorXd::Ones(Nx * Nz);
    std::vector<Triplet<double>> tripletList;

    size_t tripletSize = (Nz - 2) * (Nx - 2) * (9 + 7 + 8 + 5) + 16 + 12 * 4 + 5 * 2 * (Nz + Nx - 4 - 4); //redo
    tripletList.reserve(tripletSize);                                                                     // to do

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
                double r = (ix - 2) * hx + eps;

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
                double r = (ix - 2) * hx + eps;

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
                double r = (ix - 2) * hx + eps;

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
                double fourth = (u(iipzp) - u(iipzm)) / 2.0 / hz; //2nd order pressure
                //double fourth = (2 / 3.0 * u(iipzp) - 2 / 3.0 * u(iipzm)) / hz + (-1 / 12.0 * u(iipzp + Neq) + 1 / 12.0 * u(iipzm - Neq)) / hz; //4th order pressure

                double fifth = 0;
                double sixth = 1 / r * (u(iihxp) - u(iihxm)) / 2.0 / hx;
                double seventh = (u(iihxp) - 2 * u(iih) + u(iihxm)) / hx / hx;
                double eigth = (u(iihzp) - 2 * u(iih) + u(iihzm)) / hz / hz;

                double nlinear = first + second + third;
                double linear = fourth - 1 / lam * (fifth + sixth + seventh + eigth);

                rhs(ii) = linear + nlinear;

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
                double r = (ix - 2) * hx + eps;

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

    double dpLeft = 0;
    double dpRight = 0;
    double dpTop = 0;
    double dpBottom = 0;

    double pRef = 1;

    double cval = 1;

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

    //Corners BC
    {
        //left top corner
        size_t ix = 2;
        size_t iz = 2;

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

        rhs(iifxm) = u(iif) - fLeft;
        rhs(iigxm) = u(iig) - gLeft;
        rhs(iihxm) = u(iih) - hLeft;
        rhs(iipxm) = (u(iipxp) - u(iipxm)) / 2.0 / hx - dpLeft;

        rhs(iifzm) = (u(iipzp) - u(iipzm)) / 2.0 / hz - dpTop;
        rhs(iigzm) = (u(iifxp) - u(iifxm)) / 2.0 / hx;
        rhs(iihzm) = (u(iigxp) - u(iigxm)) / 2.0 / hx;
        rhs(iipzm) = u(iip) - pRef;

        //left
        tripletList.push_back(Triplet<double>(iifxm, iif, 1));
        tripletList.push_back(Triplet<double>(iigxm, iig, 1));
        tripletList.push_back(Triplet<double>(iihxm, iih, 1));

        tripletList.push_back(Triplet<double>(iipxm, iipxp, 1 / 2.0 / hx));
        tripletList.push_back(Triplet<double>(iipxm, iipxm, -1 / 2.0 / hx));

        //top
        tripletList.push_back(Triplet<double>(iifzm, iipzp, 1 / 2.0 / hz));
        tripletList.push_back(Triplet<double>(iifzm, iipzm, -1 / 2.0 / hz));

        tripletList.push_back(Triplet<double>(iigzm, iifxp, 1 / 2.0 / hx));
        tripletList.push_back(Triplet<double>(iigzm, iifxm, -1 / 2.0 / hx));

        tripletList.push_back(Triplet<double>(iihzm, iigxp, 1 / 2.0 / hx));
        tripletList.push_back(Triplet<double>(iihzm, iigxm, -1 / 2.0 / hx));

        tripletList.push_back(Triplet<double>(iipzm, iip, 1));
    }

    {
        //left bottom corner
        size_t ix = 2;
        size_t iz = Nz - 3;

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

        size_t ip21 = (3 + Nz + Nz) * Neq + 3;

        rhs(iifxm) = u(ip21) - pRef;
        rhs(iigxm) = u(iig) - gLeft;
        rhs(iihxm) = u(iih) - hLeft;
        rhs(iipxm) = (u(iipxp) - u(iipxm)) / 2.0 / hx - dpLeft;

        rhs(iifzp) = (u(iipzp) - u(iipzm)) / 2.0 / hz - dpBottom;
        rhs(iigzp) = (u(iifxp) - u(iifxm)) / 2.0 / hx;
        rhs(iihzp) = (u(iigxp) - u(iigxm)) / 2.0 / hx;
        rhs(iipzp) = u(iip) - pRef;

        //left
        tripletList.push_back(Triplet<double>(iifxm, ip21, 1));
        tripletList.push_back(Triplet<double>(iigxm, iig, 1));
        tripletList.push_back(Triplet<double>(iihxm, iih, 1));

        tripletList.push_back(Triplet<double>(iipxm, iipxp, 1 / 2.0 / hx));
        tripletList.push_back(Triplet<double>(iipxm, iipxm, -1 / 2.0 / hx));

        //bottom
        tripletList.push_back(Triplet<double>(iifzp, iipzp, 1 / 2.0 / hz));
        tripletList.push_back(Triplet<double>(iifzp, iipzm, -1 / 2.0 / hz));

        tripletList.push_back(Triplet<double>(iigzp, iifxp, 1 / 2.0 / hx));
        tripletList.push_back(Triplet<double>(iigzp, iifxm, -1 / 2.0 / hx));

        tripletList.push_back(Triplet<double>(iihzp, iigxp, 1 / 2.0 / hx));
        tripletList.push_back(Triplet<double>(iihzp, iigxm, -1 / 2.0 / hx));

        tripletList.push_back(Triplet<double>(iipzp, iip, 1));
    }

    {
        //right top corner
        size_t ix = Nx - 3;
        size_t iz = 2;

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

        size_t ip12 = (2 + Nz + Nz + Nz) * Neq + 3;

        rhs(iifxp) = u(iif) - fRight;
        rhs(iigxp) = u(iig) - gRight;
        rhs(iihxp) = u(ip12) - pRef;
        rhs(iipxp) = (u(iipxp) - u(iipxm)) / 2.0 / hx - dpRight;

        rhs(iifzm) = (u(iipzp) - u(iipzm)) / 2.0 / hz - dpTop;
        rhs(iigzm) = (u(iifxp) - u(iifxm)) / 2.0 / hx;
        rhs(iihzm) = (u(iigxp) - u(iigxm)) / 2.0 / hx;
        rhs(iipzm) = u(iip) - pRef;

        //right
        tripletList.push_back(Triplet<double>(iifxp, iif, 1));
        tripletList.push_back(Triplet<double>(iigxp, iig, 1));
        tripletList.push_back(Triplet<double>(iihxp, ip12, 1));

        tripletList.push_back(Triplet<double>(iipxp, iipxp, 1 / 2.0 / hx));
        tripletList.push_back(Triplet<double>(iipxp, iipxm, -1 / 2.0 / hx));

        //top
        tripletList.push_back(Triplet<double>(iifzm, iipzp, 1 / 2.0 / hz));
        tripletList.push_back(Triplet<double>(iifzm, iipzm, -1 / 2.0 / hz));

        tripletList.push_back(Triplet<double>(iigzm, iifxp, 1 / 2.0 / hx));
        tripletList.push_back(Triplet<double>(iigzm, iifxm, -1 / 2.0 / hx));

        tripletList.push_back(Triplet<double>(iihzm, iigxp, 1 / 2.0 / hx));
        tripletList.push_back(Triplet<double>(iihzm, iigxm, -1 / 2.0 / hx));

        tripletList.push_back(Triplet<double>(iipzm, iip, 1));
    }

    {
        //right bottom corner
        size_t ix = Nx - 3;
        size_t iz = Nz - 3;

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

        size_t ip22 = (1 + Nz + Nz + 1 + Nz + 3) * Neq + 3;

        rhs(iifxp) = u(ip22) - pRef;
        rhs(iigxp) = u(iig) - gRight;
        rhs(iihxp) = u(iih) - hRight;
        rhs(iipxp) = (u(iipxp) - u(iipxm)) / 2.0 / hx - dpRight;

        rhs(iifzp) = (u(iipzp) - u(iipzm)) / 2.0 / hz - dpBottom;
        rhs(iigzp) = (u(iifxp) - u(iifxm)) / 2.0 / hx;
        rhs(iihzp) = (u(iigxp) - u(iigxm)) / 2.0 / hx;
        rhs(iipzp) = u(iip) - pRef;

        //right
        tripletList.push_back(Triplet<double>(iifxp, ip22, 1));
        tripletList.push_back(Triplet<double>(iigxp, iig, 1));
        tripletList.push_back(Triplet<double>(iihxp, iih, 1));

        tripletList.push_back(Triplet<double>(iipxp, iipxp, 1 / 2.0 / hx));
        tripletList.push_back(Triplet<double>(iipxp, iipxm, -1 / 2.0 / hx));

        //bottom
        tripletList.push_back(Triplet<double>(iifzp, iipzp, 1 / 2.0 / hz));
        tripletList.push_back(Triplet<double>(iifzp, iipzm, -1 / 2.0 / hz));

        tripletList.push_back(Triplet<double>(iigzp, iifxp, 1 / 2.0 / hx));
        tripletList.push_back(Triplet<double>(iigzp, iifxm, -1 / 2.0 / hx));

        tripletList.push_back(Triplet<double>(iihzp, iigxp, 1 / 2.0 / hx));
        tripletList.push_back(Triplet<double>(iihzp, iigxm, -1 / 2.0 / hx));

        tripletList.push_back(Triplet<double>(iipzp, iip, 1));
    }

    {
        //General BC not in the corners
        size_t ieq = 0;

        // left
        for (size_t ix = 2; ix < 3; ix++)
        {
            for (size_t iz = 1 + 1 + 1; iz < Nz - 1 - 1 - 1; iz++)
            {
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

                rhs(iifxm) = u(iif) - fLeft;
                rhs(iigxm) = u(iig) - gLeft;
                rhs(iihxm) = u(iih) - hLeft;
                rhs(iipxm) = (u(iipxp) - u(iipxm)) / 2.0 / hx - dpLeft;

                tripletList.push_back(Triplet<double>(iifxm, iif, 1));
                tripletList.push_back(Triplet<double>(iigxm, iig, 1));
                tripletList.push_back(Triplet<double>(iihxm, iih, 1));

                tripletList.push_back(Triplet<double>(iipxm, iipxp, 1 / 2.0 / hx));
                tripletList.push_back(Triplet<double>(iipxm, iipxm, -1 / 2.0 / hx));
            }
        }

        // right
        for (size_t ix = Nx - 3; ix < Nx - 2; ix++)
        {
            for (size_t iz = 1 + 1 + 1; iz < Nz - 1 - 1 - 1; iz++)
            {
                double r = (ix - 2) * hx + eps;

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

                rhs(iifxp) = u(iif) - fRight;
                rhs(iigxp) = u(iig) - gRight;
                rhs(iihxp) = u(iih) - hRight;
                rhs(iipxp) = (u(iipxp) - u(iipxm)) / 2.0 / hx - dpRight;

                tripletList.push_back(Triplet<double>(iifxp, iif, 1));
                tripletList.push_back(Triplet<double>(iigxp, iig, 1));
                tripletList.push_back(Triplet<double>(iihxp, iih, 1));

                tripletList.push_back(Triplet<double>(iipxp, iipxp, 1 / 2.0 / hx));
                tripletList.push_back(Triplet<double>(iipxp, iipxm, -1 / 2.0 / hx));
            }
        }

        // top
        for (size_t ix = 1 + 1 + 1; ix < Nx - 1 - 1 - 1; ix++)
        {
            for (size_t iz = 1 + 1; iz < 2 + 1; iz++)
            {
                double r = (ix - 2) * hx + eps;

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

                rhs(iifzm) = u(iif) - fTop;
                rhs(iigzm) = u(iig) - gTop * r;
                rhs(iihzm) = u(iih) - hTop;
                rhs(iipzm) = (u(iipzp) - u(iipzm)) / 2.0 / hz - dpTop;

                tripletList.push_back(Triplet<double>(iifzm, iif, 1));
                tripletList.push_back(Triplet<double>(iigzm, iig, 1));
                tripletList.push_back(Triplet<double>(iihzm, iih, 1));

                tripletList.push_back(Triplet<double>(iipzm, iipzp, 1 / 2.0 / hz));
                tripletList.push_back(Triplet<double>(iipzm, iipzm, -1 / 2.0 / hz));
            }
        }

        // bottom
        for (size_t ix = 1 + 1 + 1; ix < Nx - 1 - 1 - 1; ix++)
        {
            for (size_t iz = Nz - 3; iz < Nz - 2; iz++)
            {
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

                rhs(iifzp) = u(iif) - fBottom;
                rhs(iigzp) = u(iig) - gBottom;
                rhs(iihzp) = u(iih) - hBottom;
                rhs(iipzp) = (u(iipzp) - u(iipzm)) / 2.0 / hz - dpBottom;

                tripletList.push_back(Triplet<double>(iifzp, iif, 1));
                tripletList.push_back(Triplet<double>(iigzp, iig, 1));
                tripletList.push_back(Triplet<double>(iihzp, iih, 1));

                tripletList.push_back(Triplet<double>(iipzp, iipzp, 1 / 2.0 / hz));
                tripletList.push_back(Triplet<double>(iipzp, iipzm, -1 / 2.0 / hz));
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
        for (size_t iz = 1; iz < 2; iz++)
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

            rhs(iifzm) = u(iifzm) - cval;
            rhs(iigzm) = u(iigzm) - cval;
            rhs(iihzm) = u(iihzm) - cval;

            rhs(iipzm) = (u(iipzp) - u(iipzm)) / 2.0 / hz - dpTop;

            tripletList.push_back(Triplet<double>(iifzm, iifzm, 1));
            tripletList.push_back(Triplet<double>(iigzm, iigzm, 1));
            tripletList.push_back(Triplet<double>(iihzm, iihzm, 1));

            tripletList.push_back(Triplet<double>(iipzm, iipzp, 1 / 2.0 / hz));
            tripletList.push_back(Triplet<double>(iipzm, iipzm, -1 / 2.0 / hz));
        }
    }

    // bot
    for (size_t ix = 1; ix < Nx - 1; ix++)
    {
        for (size_t iz = Nz - 1 - 1; iz < Nz - 1; iz++)
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

            rhs(iifzp) = u(iifzp) - cval;
            rhs(iigzp) = u(iigzp) - cval;
            rhs(iihzp) = u(iihzp) - cval;

            rhs(iipzp) = (u(iipzp) - u(iipzm)) / 2.0 / hz - dpBottom;

            tripletList.push_back(Triplet<double>(iifzp, iifzp, 1));
            tripletList.push_back(Triplet<double>(iigzp, iigzp, 1));
            tripletList.push_back(Triplet<double>(iihzp, iihzp, 1));

            tripletList.push_back(Triplet<double>(iipzp, iipzp, 1 / 2.0 / hz));
            tripletList.push_back(Triplet<double>(iipzp, iipzm, -1 / 2.0 / hz));
        }
    }

    //left
    for (size_t ix = 1; ix < 2; ix++)
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

            rhs(iifxm) = u(iifxm) - cval;
            rhs(iigxm) = u(iigxm) - cval;
            rhs(iihxm) = u(iihxm) - cval;

            rhs(iipxm) = (u(iipxp) - u(iipxm)) / 2.0 / hx - dpLeft;

            tripletList.push_back(Triplet<double>(iifxm, iifxm, 1));
            tripletList.push_back(Triplet<double>(iigxm, iigxm, 1));
            tripletList.push_back(Triplet<double>(iihxm, iihxm, 1));

            tripletList.push_back(Triplet<double>(iipxm, iipxp, 1 / 2.0 / hx));
            tripletList.push_back(Triplet<double>(iipxm, iipxm, -1 / 2.0 / hx));
        }
    }

    //right
    for (size_t ix = Nx - 2; ix < Nx - 1; ix++)
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

            rhs(iifxp) = u(iifxp) - cval;
            rhs(iigxp) = u(iigxp) - cval;
            rhs(iihxp) = u(iihxp) - cval;

            rhs(iipxp) = (u(iipxp) - u(iipxm)) / 2.0 / hx - dpRight;

            tripletList.push_back(Triplet<double>(iifxp, iifxp, 1));
            tripletList.push_back(Triplet<double>(iigxp, iigxp, 1));
            tripletList.push_back(Triplet<double>(iihxp, iihxp, 1));

            tripletList.push_back(Triplet<double>(iipxp, iipxp, 1 / 2.0 / hx));
            tripletList.push_back(Triplet<double>(iipxp, iipxm, -1 / 2.0 / hx));
        }
    }

    // for (size_t i = 0; i < tripletList.size(); i++)
    // {
    //     std::cout << tripletList[i].row() << " " << tripletList[i].col() << " " << tripletList[i].value() << std::endl;
    // }
    // return;
    //internal::BandMatrix<double> a(Nx*Nz*Neq,Nx*Nz*Neq,13,13);
    //internal::BandMatrix<double> a(5,5,1,1);

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
    //std::cout << "init:" << duration2.count() << std::endl;

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
    //std::cout << "solve:" << duration.count() << std::endl;

    u = u + du;
    {
        double res = (jac * du + rhs).norm();
        std::cout << "1er Res is:" << res << std::endl;
    }
    {
        double res = (du).norm();
        std::cout << "2eme Res is:" << res << std::endl;
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
            int ii = (iz - 2) + (Nx - 4) * (ix - 2);
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
