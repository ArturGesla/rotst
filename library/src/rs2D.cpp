#include "rs2D.hpp"
#include "lean_vtk.hpp"

#include <Eigen/Eigenvalues>

#include <chrono>

using namespace std::chrono;

void rs2D()
{
    // there is a problem with r=0, 1/r

    size_t nx = 3; // with bc without ghost
    size_t nz = 3; // with bc without ghost

    size_t Neq = 4;

    size_t Nx = nx + 2;
    size_t Nz = nz + 2;

    double Lx = 1;
    double Lz = 1;

    double hx = Lx / double(nx - 1);
    double hz = Lz / double(nz - 1);

    double lam = 1;

    VectorXd rhs = VectorXd::Zero(Nx * Nz * Neq);
    SparseMatrix<double> jac(Nx * Nz * Neq, Nx * Nz * Neq);

    // VectorXd f = VectorXd::Ones(Nx * Nz);
    // VectorXd g = VectorXd::Ones(Nx * Nz);
    // VectorXd h = VectorXd::Ones(Nx * Nz);
    // VectorXd p = VectorXd::Ones(Nx * Nz);
    VectorXd u = VectorXd::Ones(Nx * Nz * Neq);
    std::vector<Triplet<double>> tripletList;
    tripletList.reserve((Nx - 1) * (Nz - 1) * (7 + 5 + 7 + 5)); // to do

    // std::cout << "expected size: " << (Nx - 2) * (Nz - 2) * (7 + 5 + 7 + 5) << std::endl;
    //  test
    //   std::vector<double> v1 = {1.0, 2.0, 3.0};
    //   Eigen::Vector3d v2(v1.data());
    //   std::cout<<v2<<std::endl;

    {
        // x momentum
        size_t ieq = 0;
        for (size_t ix = 1; ix < Nx - 1; ix++)
        {
            for (size_t iz = 1; iz < Nz - 1; iz++)
            {
                double r = ix * hx;

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

                double first;
                double second;
                double third;
                double fourth = (u(iipxp) - u(iipxm)) / 2.0 / hx;
                double fifth = -u(iif) / r / r;
                double sixth = 1 / r * (u(iifxp) - u(iifxm)) / 2.0 / hx;
                double seventh = (u(iifxp) - 2 * u(iif) + u(iifxm)) / hx / hx;
                double eigth = (u(iifzp) - 2 * u(iif) + u(iifzm)) / hz / hz;

                double nlinear;
                double linear = fourth - 1 / lam * (fifth + sixth + seventh + eigth);

                rhs(ii) = linear;

                tripletList.push_back(Triplet<double>(ii, iipxp, 1 / 2.0 / hx));
                tripletList.push_back(Triplet<double>(ii, iipxm, -1 / 2.0 / hx));
                tripletList.push_back(Triplet<double>(ii, iif, -1 / lam * (-1 / r / r - 2 / hx / hx - 2 / hz / hz)));
                tripletList.push_back(Triplet<double>(ii, iifxp, -1 / lam * (1 / r / 2.0 / hx + 1 / hx / hx)));
                tripletList.push_back(Triplet<double>(ii, iifxm, -1 / lam * (-1 / r / 2.0 / hx + 1 / hx / hx)));
                tripletList.push_back(Triplet<double>(ii, iifzp, -1 / lam * (1 / hz / hz)));
                tripletList.push_back(Triplet<double>(ii, iifzm, -1 / lam * (1 / hz / hz)));
            }
        }
    }

    {
        // y momentum
        size_t ieq = 1;
        for (size_t ix = 1; ix < Nx - 1; ix++)
        {
            for (size_t iz = 1; iz < Nz - 1; iz++)
            {
                double r = ix * hx;

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

                double first;
                double second;
                double third;
                double fourth = 0;
                double fifth = -u(iig) / r / r;
                double sixth = 1 / r * (u(iigxp) - u(iigxm)) / 2.0 / hx;
                double seventh = (u(iigxp) - 2 * u(iig) + u(iigxm)) / hx / hx;
                double eigth = (u(iigzp) - 2 * u(iig) + u(iigzm)) / hz / hz;

                double nlinear;
                double linear = fourth - 1 / lam * (fifth + sixth + seventh + eigth);

                rhs(ii) = linear;

                tripletList.push_back(Triplet<double>(ii, iig, -1 / lam * (-1 / r / r - 2 / hx / hx - 2 / hz / hz)));
                tripletList.push_back(Triplet<double>(ii, iigxp, -1 / lam * (1 / r / 2.0 / hx + 1 / hx / hx)));
                tripletList.push_back(Triplet<double>(ii, iigxm, -1 / lam * (-1 / r / 2.0 / hx + 1 / hx / hx)));
                tripletList.push_back(Triplet<double>(ii, iigzp, -1 / lam * (1 / hz / hz)));
                tripletList.push_back(Triplet<double>(ii, iigzm, -1 / lam * (1 / hz / hz)));
            }
        }
    }

    {
        // z momentum
        size_t ieq = 2;
        for (size_t ix = 1; ix < Nx - 1; ix++)
        {
            for (size_t iz = 1; iz < Nz - 1; iz++)
            {
                double r = ix * hx;

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

                double first;
                double second;
                double third;
                double fourth = (u(iipzp) - u(iipzm)) / 2.0 / hz;
                double fifth = 0;
                double sixth = 1 / r * (u(iihxp) - u(iihxm)) / 2.0 / hx;
                double seventh = (u(iihxp) - 2 * u(iih) + u(iihxm)) / hx / hx;
                double eigth = (u(iihzp) - 2 * u(iih) + u(iihzm)) / hz / hz;

                double nlinear;
                double linear = fourth - 1 / lam * (fifth + sixth + seventh + eigth);

                rhs(ii) = linear;

                tripletList.push_back(Triplet<double>(ii, iipzp, 1 / 2.0 / hz));
                tripletList.push_back(Triplet<double>(ii, iipzm, -1 / 2.0 / hz));
                tripletList.push_back(Triplet<double>(ii, iih, -1 / lam * (-2 / hx / hx - 2 / hz / hz)));
                tripletList.push_back(Triplet<double>(ii, iihxp, -1 / lam * (1 / r / 2.0 / hx + 1 / hx / hx)));
                tripletList.push_back(Triplet<double>(ii, iihxm, -1 / lam * (-1 / r / 2.0 / hx + 1 / hx / hx)));
                tripletList.push_back(Triplet<double>(ii, iihzp, -1 / lam * (1 / hz / hz)));
                tripletList.push_back(Triplet<double>(ii, iihzm, -1 / lam * (1 / hz / hz)));
            }
        }
    }

    {
        // conti
        size_t ieq = 3;
        for (size_t ix = 1; ix < Nx - 1; ix++)
        {
            for (size_t iz = 1; iz < Nz - 1; iz++)
            {
                double r = ix * hx;

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

    double cval = 100;

    {
        // 4 corner points are constant for all the BC
        for (size_t ieq = 0; ieq < Neq; ieq++)
        {
            size_t i_left_top = 0 + ieq;
            size_t i_right_top = (Nx * Nz - Nz) * Neq + ieq;
            size_t i_left_bot = (Nz - 1) * Neq + ieq;
            size_t i_right_bot = (Nx * Nz - 1) * Neq + ieq;

            rhs(i_left_top) = u(i_left_top) - cval;
            rhs(i_right_top) = u(i_right_top) - cval;
            rhs(i_left_bot) = u(i_left_bot) - cval;
            rhs(i_right_bot) = u(i_right_bot) - cval;

            tripletList.push_back(Triplet<double>(i_left_top, i_left_top, 1));
            tripletList.push_back(Triplet<double>(i_right_top, i_right_top, 1));
            tripletList.push_back(Triplet<double>(i_left_bot, i_left_bot, 1));
            tripletList.push_back(Triplet<double>(i_right_bot, i_right_bot, 1));
        }
    }

    {
        // additional points in corners only for Dirichlet BC for vel - 12 jacobian entries
        std::array<int, 4> pts = {Nz + 0,
                                  Nz + Nz - 1,
                                  -Nz + Nx * Nz - Nz,
                                  -Nz + Nx * Nz - 1};

        // velocity
        // for (size_t ieq = 0; ieq < Neq - 1; ieq++)
        // {
        //     for (size_t i = 0; i < pts.max_size(); i++)
        //     {
        //         size_t j = pts[i] * Neq + ieq;
        //         rhs(j) = u(j) - cval;
        //         tripletList.push_back(Triplet<double>(j, j, 1));
        //     }
        // }

        // pressure
        // size_t ieq = 3;
        // {
        //     // add point 0
        //     size_t i = 0;
        //     size_t j = pts[i] * Neq + ieq;
        //     rhs(j) = (u(j + 2 * Neq) - u(j)) / 2.0 / hz - dpTop;
        //     tripletList.push_back(Triplet<double>(j, j + 2 * Neq, 1 / 2.0 / hz));
        //     tripletList.push_back(Triplet<double>(j, j, -1 / 2.0 / hz));
        // }
        // {
        //     // add point 1
        //     size_t i = 1;
        //     size_t j = pts[i] * Neq + ieq;
        //     rhs(j) = (u(j) - u(j - 2 * Neq)) / 2.0 / hz - dpBottom;
        //     tripletList.push_back(Triplet<double>(j, j - 2 * Neq, -1 / 2.0 / hz));
        //     tripletList.push_back(Triplet<double>(j, j, 1 / 2.0 / hz));
        // }
        // {
        //     // add point 2
        //     size_t i = 2;
        //     size_t j = pts[i] * Neq + ieq;
        //     rhs(j) = (u(j + 2 * Neq) - u(j)) / 2.0 / hz - dpTop;
        //     tripletList.push_back(Triplet<double>(j, j + 2 * Neq, 1 / 2.0 / hz));
        //     tripletList.push_back(Triplet<double>(j, j, -1 / 2.0 / hz));
        // }
        // {
        //     // add point 3
        //     size_t i = 3;
        //     size_t j = pts[i] * Neq + ieq;
        //     rhs(j) = u(j) - pRef;
        //     tripletList.push_back(Triplet<double>(j, j, 1));
        // }

        // podejscie 2 - 4 wartosci cisnienia 1 2 3 4
        // size_t ieq = 3;
        // {
        //     // add point 0
        //     size_t i = 0;
        //     size_t j = pts[i] * Neq + ieq;
        //     rhs(j) = u(j) - 1;
        //     tripletList.push_back(Triplet<double>(j, j, 1));
        // }
        // {
        //     // add point 1
        //     size_t i = 1;
        //     size_t j = pts[i] * Neq + ieq;
        //     rhs(j) = u((Nz + 1) * Neq + ieq) - 2;
        //     tripletList.push_back(Triplet<double>(j, (Nz + 1) * Neq + ieq, 1));
        // }
        // {
        //     // add point 2
        //     size_t i = 2;
        //     size_t j = pts[i] * Neq + ieq;
        //     rhs(j) = u((Nz + Nz) * Neq + ieq) - 3;
        //     tripletList.push_back(Triplet<double>(j, (Nz + Nz) * Neq + ieq, 1));
        // }
        // {
        //     // add point 3
        //     size_t i = 3;
        //     size_t j = pts[i] * Neq + ieq;
        //     rhs(j) = u((Nz + Nz + 1) * Neq + ieq) - 4;
        //     tripletList.push_back(Triplet<double>(j, (Nz + Nz + 1) * Neq + ieq, 1));
        // }
    }

    {
        size_t ieq = 0;

        // left
        for (size_t ix = 1; ix < 2; ix++)
        {
            for (size_t iz = 1; iz < Nz - 1; iz++)
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
        for (size_t ix = Nx - 2; ix < Nx - 1; ix++)
        {
            for (size_t iz = 1; iz < Nz - 1; iz++)
            {
                double r = ix * hx;

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
                rhs(iigxp) = u(iig) - gRight * r;
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
        for (size_t ix = 1 + 1; ix < Nx - 1 - 1; ix++)
        {
            for (size_t iz = 1; iz < 2; iz++)
            {
                double r = ix * hx;

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
        for (size_t ix = 1 + 1; ix < Nx - 1 - 1; ix++)
        {
            for (size_t iz = Nz - 2; iz < Nz - 1; iz++)
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

    jac.setFromTriplets(tripletList.begin(), tripletList.end());

    for (size_t i = 0; i < tripletList.size(); i++)
    {
        std::cout << tripletList[i].row() << " " << tripletList[i].col() << " " << tripletList[i].value() << std::endl;
    }

    // std::cout << jac << std::endl;

    // std::cout << rhs << std::endl;
    // std::cout << jac << std::endl;
    // std::cout << "rl size: " << tripletList.size() << std::endl;

    SparseLU<SparseMatrix<double>> solver;
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

    u = u + du;
    double res = (jac * u + rhs).norm();
    std::cout << "Res is:" << res << std::endl;

    // writeVTU_vp(hx, Nx, hz, Nz, u);
}

void writeVTU_vp(double hx, double Nx, double hz, double Nz, VectorXd u)
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
    std::string filename = "single_tri.vtu";
    leanvtk::VTUWriter writer;

    // my data
    std::vector<double> points2;
    std::vector<int> elements2;
    std::vector<double> scalar_field2;

    for (size_t ix = 1; ix < Nx - 1; ix++)
    {
        for (size_t iz = 1; iz < Nz - 1; iz++)
        {
            double z = 0;
            int ii = iz + Nz * ix;

            points2.push_back((ix - 1) * hx);
            points2.push_back((iz - 1) * hz);
            points2.push_back(z);

            scalar_field2.push_back(u(ii));
        }
    }

    for (size_t ix = 1; ix < Nx - 1 - 1; ix++)
    {
        for (size_t iz = 1; iz < Nz - 1 - 1; iz++)
        {
            double z = 0;
            int ii = (iz - 1) + (Nx - 2) * (ix - 1);
            // myfile << ix * hx << ",\t\t" << (Nz - 2 - iz) * hz << ",\t\t" << z << ",\t\t" << phi(ii) << std::endl;

            elements2.push_back(ii);
            elements2.push_back(ii + 1);
            elements2.push_back(ii + 1 + Nz - 2);
            elements2.push_back(ii + 1 + Nz - 3);
        }
    }

    writer.add_scalar_field("scalar_field", scalar_field2);
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
