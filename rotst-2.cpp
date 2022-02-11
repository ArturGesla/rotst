#include "rotst-2.hpp"

#include <vector>
#include <fstream>
#include <chrono>

#include <Eigen/Sparse>
#include <Eigen/SparseQR>

#include "lean_vtk.hpp"

using namespace Eigen::placeholders;
using namespace std::chrono;

void test()
{
    rs1D();
}

void diff2D()
{
    int nx = 30; //with bc without ghost
    int ny = 30; //with bc without ghost

    int Nx = nx + 2;
    int Ny = ny + 2;

    double Lx = 10;
    double Ly = 10;

    double hx = Lx / (nx - 1);
    double hy = Ly / (ny - 1);

    VectorXd rhs = VectorXd::Zero(Nx * Ny);
    SparseMatrix<double> jac(Nx * Ny, Nx * Ny);

    VectorXd phi = VectorXd::Zero(Nx * Ny);

    //interior
    for (size_t ix = 1; ix < Nx - 1; ix++)
    {
        for (size_t iy = 1; iy < Ny - 1; iy++)
        {
            int ii = iy + ix * Nx;
            int iixm = ii - Nx;
            int iixp = ii + Nx;
            int iiym = ii - 1;
            int iiyp = ii + 1;

            rhs(ii) = (phi(iixp) - 2 * phi(ii) + phi(iixm)) / hx / hx + (phi(iiyp) - 2 * phi(ii) + phi(iiym)) / hy / hy;

            jac.insert(ii, iixp) = 1 / hx / hx;
            jac.insert(ii, ii) = -2 / hx / hx - 2 / hy / hy;
            jac.insert(ii, iixm) = 1 / hx / hx;
            jac.insert(ii, iiyp) = 1 / hy / hy;
            jac.insert(ii, iiym) = 1 / hy / hy;
        }
    }

    //4 corner points
    double cval = 100;

    rhs(0) = phi(0) - cval;
    rhs(Ny - 1) = phi(Ny - 1) - cval;
    rhs(Nx * Ny - Ny) = phi(Nx * Ny - Ny) - cval;
    rhs(Nx * Ny - 1) = phi(Nx * Ny - 1) - cval;

    jac.insert(0, 0) = 1;
    jac.insert(Ny - 1, Ny - 1) = 1;
    jac.insert(Nx * Ny - Ny, Nx * Ny - Ny) = 1;
    jac.insert(Nx * Ny - 1, Nx * Ny - 1) = 1;

    //additional points in corners
    std::array<int, 4> pts = {Ny + 0,
                              Ny + Ny - 1,
                              -Ny + Nx * Ny - Ny,
                              -Ny + Nx * Ny - 1};

    /*     rhs(Ny + 0) = phi(Ny + 0) - cval;
    rhs(Ny + Ny - 1) = phi(Ny + Ny - 1) - cval;
    rhs(-Ny + Nx * Ny - Ny) = phi(-Ny + Nx * Ny - Ny) - cval;
    rhs(-Ny + Nx * Ny - 1) = phi(-Ny + Nx * Ny - 1) - cval;

    jac.insert(0, 0) = 1;
    jac.insert(Ny - 1, Ny - 1) = 1;
    jac.insert(Nx * Ny - Ny, Nx * Ny - Ny) = 1;
    jac.insert(-Ny + Nx * Ny - 1, -Ny + Nx * Ny - 1) = 1;
     */

    for (size_t i = 0; i < pts.max_size(); i++)
    {
        Index j = pts[i];
        rhs(j) = phi(j) - cval;
        jac.insert(j, j) = 1;
    }

    //Dirichlet BC on walls
    double dbcLeft = 5.0;
    double dbcRight = 10.0;
    double dbcTop = -5.0;
    double dbcBot = -10.0;

    //left
    for (size_t ix = 1; ix < 2; ix++)
    {
        for (size_t iy = 1; iy < Ny - 1; iy++)
        {
            int ii = iy + ix * Nx;
            int iixm = ii - Nx;
            int iixp = ii + Nx;
            int iiym = ii - 1;
            int iiyp = ii + 1;

            rhs(iixm) = phi(ii) - dbcLeft;

            jac.insert(iixm, ii) = 1;
        }
    }

    //right
    for (size_t ix = Nx - 2; ix < Nx - 1; ix++)
    {
        for (size_t iy = 1; iy < Ny - 1; iy++)
        {
            int ii = iy + ix * Nx;
            int iixm = ii - Nx;
            int iixp = ii + Nx;
            int iiym = ii - 1;
            int iiyp = ii + 1;

            rhs(iixp) = phi(ii) - dbcRight;
            jac.insert(iixp, ii) = 1;
        }
    }

    //top
    for (size_t ix = 1 + 1; ix < Nx - 1 - 1; ix++)
    {
        for (size_t iy = 1; iy < 2; iy++)
        {
            int ii = iy + ix * Nx;
            int iixm = ii - Nx;
            int iixp = ii + Nx;
            int iiym = ii - 1;
            int iiyp = ii + 1;

            rhs(iiym) = phi(ii) - dbcTop;

            jac.insert(iiym, ii) = 1;
        }
    }

    //bot
    for (size_t ix = 1 + 1; ix < Nx - 1 - 1; ix++)
    {
        for (size_t iy = Ny - 2; iy < Ny - 1; iy++)
        {
            int ii = iy + ix * Nx;
            int iixm = ii - Nx;
            int iixp = ii + Nx;
            int iiym = ii - 1;
            int iiyp = ii + 1;

            rhs(iiyp) = phi(ii) - dbcBot;

            jac.insert(iiyp, ii) = 1;
        }
    }

    //std::cout << rhs << std::endl;
    //std::cout << jac << std::endl;

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

    phi = phi + du;
    double res = (jac * phi + rhs).norm();
    std::cout << "Res is:" << res << std::endl;

    std::ofstream myfile;
    myfile.open("outdata.csv");

    myfile << "x,\t\ty,\t\tz,\t\tphi" << std::endl;

    for (size_t ix = 1; ix < Nx - 1; ix++)
    {
        for (size_t iy = 1; iy < Ny - 1; iy++)
        {
            double z = 0;
            int ii = iy + Nx * ix;
            myfile << ix * hx << ",\t\t" << (Ny - 2 - iy) * hy << ",\t\t" << z << ",\t\t" << phi(ii) << std::endl;
        }
    }

    myfile.close();

    writeVTU(hx, Nx, hy, Ny, phi);
}

void writeVTU(double hx, double Nx, double hy, double Ny, VectorXd phi)
{
    //his data
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

    //my data
    std::vector<double> points2;
    std::vector<int> elements2;
    std::vector<double> scalar_field2;

    for (size_t ix = 1; ix < Nx - 1; ix++)
    {
        for (size_t iy = 1; iy < Ny - 1; iy++)
        {
            double z = 0;
            int ii = iy + Nx * ix;
            //myfile << ix * hx << ",\t\t" << (Ny - 2 - iy) * hy << ",\t\t" << z << ",\t\t" << phi(ii) << std::endl;

            points2.push_back((ix - 1) * hx);
            points2.push_back((Ny - 2 - iy) * hy);
            points2.push_back(z);

            scalar_field2.push_back(phi(ii));
        }
    }

    for (size_t ix = 1; ix < Nx - 1 - 1; ix++)
    {
        for (size_t iy = 1; iy < Ny - 1 - 1; iy++)
        {
            double z = 0;
            int ii = (iy - 1) + (Nx - 2) * (ix - 1);
            //myfile << ix * hx << ",\t\t" << (Ny - 2 - iy) * hy << ",\t\t" << z << ",\t\t" << phi(ii) << std::endl;

            elements2.push_back(ii);
            elements2.push_back(ii + 1);
            elements2.push_back(ii + 1 + Ny - 2);
            elements2.push_back(ii + 1 + Ny - 3);
        }
    }

    writer.add_scalar_field("scalar_field", scalar_field2);
    //writer.add_vector_field("vector_field", vector_field, dim);

    //writer.write_point_cloud(filename, dim, points2);
    writer.write_surface_mesh(filename, dim, cell_size, points2, elements2);
}

void rs1D()
{
    auto start = high_resolution_clock::now();

    int N = 801; //with bc
    VectorXd u = VectorXd::Ones(N * 3 + 1);
    double lam = 1;
    double h = 1 / double(N - 1);
    std::array<double, 6> bc = {0, 1, 0, 0, 0, 0};

    int max_it = 4;
    std::pair<VectorXd, double> sol;

    for (size_t ir = 0; ir < 816; ir++)
    {
        lam += 1000 / 816;

        for (size_t i = 0; i < max_it; i++)
        {
            sol = newtonIteration(u, lam, h, bc);

            u = sol.first;
            double res = sol.second;

            std::cout << i << ": " << res << std::endl;
        }
    }

    save(u, h);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "whole:" << duration.count() << std::endl;
}
void convergence()
{
    double u1, u2, u3;
    double h1, h2, h3;

    {
        int N = 101; //with bc
        VectorXd u = VectorXd::Zero(N * 3 + 1);
        double lam = 625;
        double h = 1 / double(N - 1);
        std::array<double, 6> bc = {0, 1, 0, 0, 0, 0};

        int max_it = 15;

        for (size_t i = 0; i < max_it; i++)
        {
            std::pair<VectorXd, double> sol;
            sol = newtonIteration(u, lam, h, bc);

            u = sol.first;
            double res = sol.second;

            std::cout << i << ": " << res << std::endl;
        }

        u1 = u(last);
        h1 = h;
    }

    {
        int N = 201; //with bc
        VectorXd u = VectorXd::Zero(N * 3 + 1);
        double lam = 625;
        double h = 1 / double(N - 1);
        std::array<double, 6> bc = {0, 1, 0, 0, 0, 0};

        int max_it = 25;

        for (size_t i = 0; i < max_it; i++)
        {
            std::pair<VectorXd, double> sol;
            sol = newtonIteration(u, lam, h, bc);

            u = sol.first;
            double res = sol.second;

            std::cout << i << ": " << res << std::endl;
        }

        u2 = u(last);
        h2 = h;
    }

    {
        int N = 401; //with bc
        VectorXd u = VectorXd::Zero(N * 3 + 1);
        double lam = 625;
        double h = 1 / double(N - 1);
        std::array<double, 6> bc = {0, 1, 0, 0, 0, 0};

        int max_it = 25;

        for (size_t i = 0; i < max_it; i++)
        {
            std::pair<VectorXd, double> sol;
            sol = newtonIteration(u, lam, h, bc);

            u = sol.first;
            double res = sol.second;

            std::cout << i << ": " << res << std::endl;
        }

        u3 = u(last);
        h3 = h;
    }

    std::cout << u1 << "\t" << u2 << "\t" << u3 << std::endl;
    std::cout << h1 << "\t" << h2 << "\t" << h3 << std::endl;
    //std::cout << u << std::endl;
    //std::cout << res << std::endl;
    std::cout << "order: " << std::log((u2 - u1) / (u3 - u2)) / std::log(2.0) << std::endl;
    //save(u, h);
}
void save(VectorXd u, double h)
{
    using namespace Eigen::placeholders;

    VectorXd F = u(seq(0, last - 1, 3));
    VectorXd G = u(seq(1, last - 1, 3));
    VectorXd H = u(seq(2, last - 1, 3));
    double k = u(last);

    int N = F.rows();

    std::ofstream myfile;
    myfile.open("outdata");

    for (size_t i = 0; i < N; i++)
    {
        myfile << i * h << "\t\t" << F(i) << "\t\t" << G(i) << "\t\t" << H(i) << "\t\t" << k << std::endl;
    }

    myfile.close();
}
std::pair<VectorXd, double> newtonIteration(VectorXd u, double lam, double h, std::array<double, 6> bc)
{
    auto start = high_resolution_clock::now();

    using namespace Eigen::placeholders;

    VectorXd F = u(seq(0, last - 1, 3));
    VectorXd G = u(seq(1, last - 1, 3));
    VectorXd H = u(seq(2, last - 1, 3));
    double k = u(last);

    VectorXd rhs = VectorXd::Zero(u.rows());
    int N = F.rows();
    int NN = u.rows();

    SparseMatrix<double> jac(NN, NN);
    //jac.reserve(VectorXd::Constant(NN,6));

    std::vector<Triplet<double>> tripletList;
    tripletList.reserve(3 + 3 + N * (4) + (N - 1) * (6 + 5));

    //left boundary
    rhs(0) = F(0) - bc[0];
    rhs(1) = G(0) - bc[1];
    rhs(2) = H(0) - bc[2];

    //jac.insert(0, 0) = 1;
    //jac.insert(1, 1) = 1;
    //jac.insert(2, 2) = 1;

    tripletList.push_back(Triplet<double>(0, 0, 1));
    tripletList.push_back(Triplet<double>(1, 1, 1));
    tripletList.push_back(Triplet<double>(2, 2, 1));

    //this jacobian calc can be optimised

    //conti part
    for (size_t i = 0; i < N - 1; i++)
    {
        int ii = 3 + i * 3;
        rhs(ii) = F(i) + F(i + 1) + (H(i + 1) - H(i)) / h;

        int iif = i * 3;
        int iifp = i * 3 + 3;
        int iih = i * 3 + 2;
        int iihp = i * 3 + 2 + 3;
        //jac.insert(ii, iif) = 1.0;
        //jac.insert(ii, iifp) = 1.0;
        //jac.insert(ii, iih) = -1.0 / h;
        //jac.insert(ii, iihp) = 1.0 / h;

        tripletList.push_back(Triplet<double>(ii, iif, 1));
        tripletList.push_back(Triplet<double>(ii, iifp, 1));
        tripletList.push_back(Triplet<double>(ii, iih, -1 / h));
        tripletList.push_back(Triplet<double>(ii, iihp, 1 / h));

        //std::cout<<ii<<std::endl;
    }

    //F part
    for (size_t i = 1; i < N - 1; i++)
    {
        int ii = 3 + 1 + (i - 1) * 3;
        double dd = (F(i + 1) - 2 * F(i) + F(i - 1)) / h / h;
        double nl = -lam * (k + F(i) * F(i) - G(i) * G(i) + H(i) * (F(i + 1) - F(i - 1)) / 2.0 / h);
        rhs(ii) = dd + nl;

        int iif = i * 3;
        int iifp = i * 3 + 3;
        int iifm = i * 3 - 3;
        int iih = i * 3 + 2;
        int iig = i * 3 + 1;
        int iik = NN - 1;
        //jac.insert(ii, iif) = -2.0 / h / h - lam * 2 * F(i);
        //jac.insert(ii, iifp) = 1 / h / h - lam * H(i) / 2.0 / h;
        //jac.insert(ii, iifm) = 1 / h / h + lam * H(i) / 2.0 / h;
        //jac.insert(ii, iih) = -lam * (F(i + 1) - F(i - 1)) / 2.0 / h;
        //jac.insert(ii, iig) = -lam * (-2.0 * G(i));
        //jac.insert(ii, iik) = -lam;

        tripletList.push_back(Triplet<double>(ii, iif, -2.0 / h / h - lam * 2 * F(i)));
        tripletList.push_back(Triplet<double>(ii, iifp, 1 / h / h - lam * H(i) / 2.0 / h));
        tripletList.push_back(Triplet<double>(ii, iifm, 1 / h / h + lam * H(i) / 2.0 / h));
        tripletList.push_back(Triplet<double>(ii, iih, -lam * (F(i + 1) - F(i - 1)) / 2.0 / h));
        tripletList.push_back(Triplet<double>(ii, iig, -lam * (-2.0 * G(i))));
        tripletList.push_back(Triplet<double>(ii, iik, -lam));

        //std::cout << ii << std::endl;
    }

    //G part
    for (size_t i = 1; i < N - 1; i++)
    {
        int ii = 3 + 1 + 1 + (i - 1) * 3;
        double dd = (G(i + 1) - 2 * G(i) + G(i - 1)) / h / h;
        double nl = -lam * (2 * F(i) * G(i) + H(i) * (G(i + 1) - G(i - 1)) / 2.0 / h);
        rhs(ii) = dd + nl;

        //std::cout<<ii<<std::endl;

        int iig = i * 3 + 1;
        int iigp = i * 3 + 3 + 1;
        int iigm = i * 3 - 3 + 1;
        int iif = i * 3;
        int iih = i * 3 + 2;
        //jac.insert(ii, iig) = -2.0 / h / h - lam * 2 * F(i);
        //jac.insert(ii, iigp) = 1 / h / h - lam * H(i) / 2.0 / h;
        //jac.insert(ii, iigm) = 1 / h / h + lam * H(i) / 2.0 / h;
        //jac.insert(ii, iih) = -lam * (G(i + 1) - G(i - 1)) / 2.0 / h;
        //jac.insert(ii, iif) = -lam * (2.0 * G(i));

        tripletList.push_back(Triplet<double>(ii, iig, -2.0 / h / h - lam * 2 * F(i)));
        tripletList.push_back(Triplet<double>(ii, iigp, 1 / h / h - lam * H(i) / 2.0 / h));
        tripletList.push_back(Triplet<double>(ii, iigm, 1 / h / h + lam * H(i) / 2.0 / h));
        tripletList.push_back(Triplet<double>(ii, iih, -lam * (G(i + 1) - G(i - 1)) / 2.0 / h));
        tripletList.push_back(Triplet<double>(ii, iif, -lam * (2.0 * G(i))));

        //std::cout << ii << std::endl;
    }

    //right boundary
    rhs(NN - 3) = F(N - 1) - bc[3];
    rhs(NN - 2) = G(N - 1) - bc[4];
    rhs(NN - 1) = H(N - 1) - bc[5];

    //jac.insert(NN - 3, NN - 4) = 1;
    //jac.insert(NN - 2, NN - 3) = 1;
    //jac.insert(NN - 1, NN - 2) = 1;

    tripletList.push_back(Triplet<double>(NN - 3, NN - 4, 1));
    tripletList.push_back(Triplet<double>(NN - 2, NN - 3, 1));
    tripletList.push_back(Triplet<double>(NN - 1, NN - 2, 1));

    jac.setFromTriplets(tripletList.begin(), tripletList.end());

    //std::cout << jac << std::endl;

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    auto start2 = high_resolution_clock::now();

    BiCGSTAB<SparseMatrix<double>,IncompleteLUT<double>> solver; //does not work for zero ig
    //SparseLU<SparseMatrix<double>> solver;
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

    double res = rhs.norm();

    auto stop2 = high_resolution_clock::now();
    auto duration2 = duration_cast<microseconds>(stop2 - start2);

    std::cout << "init:" << duration.count() << std::endl;
    std::cout << "solve:" << duration2.count() << std::endl;

    std::pair<VectorXd, double> out = {du + u, res};

    //std::cout<<jac<<std::endl;

    return out;
}