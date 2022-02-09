#include "rotst-2.hpp"

#include <vector>
#include <fstream>

#include <Eigen/Sparse>
#include <Eigen/SparseQR>

using namespace Eigen::placeholders;

void test()
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
    std::cout << "order: " << std::log((u2 - u1)/(u3 - u2))/std::log(2.0) << std::endl;
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
    using namespace Eigen::placeholders;

    VectorXd F = u(seq(0, last - 1, 3));
    VectorXd G = u(seq(1, last - 1, 3));
    VectorXd H = u(seq(2, last - 1, 3));
    double k = u(last);

    VectorXd rhs = VectorXd::Zero(u.rows());
    int N = F.rows();
    int NN = u.rows();

    SparseMatrix<double> jac(NN, NN);

    //left boundary
    rhs(0) = F(0) - bc[0];
    rhs(1) = G(0) - bc[1];
    rhs(2) = H(0) - bc[2];

    jac.insert(0, 0) = 1;
    jac.insert(1, 1) = 1;
    jac.insert(2, 2) = 1;

    //conti part
    for (size_t i = 0; i < N - 1; i++)
    {
        int ii = 3 + i * 3;
        rhs(ii) = F(i) + F(i + 1) + (H(i + 1) - H(i)) / h;

        int iif = i * 3;
        int iifp = i * 3 + 3;
        int iih = i * 3 + 2;
        int iihp = i * 3 + 2 + 3;
        jac.insert(ii, iif) = 1.0;
        jac.insert(ii, iifp) = 1.0;
        jac.insert(ii, iih) = -1.0 / h;
        jac.insert(ii, iihp) = 1.0 / h;

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
        jac.insert(ii, iif) = -2.0 / h / h - lam * 2 * F(i);
        jac.insert(ii, iifp) = 1 / h / h - lam * H(i) / 2.0 / h;
        jac.insert(ii, iifm) = 1 / h / h + lam * H(i) / 2.0 / h;
        jac.insert(ii, iih) = -lam * (F(i + 1) - F(i - 1)) / 2.0 / h;
        jac.insert(ii, iig) = -lam * (-2.0 * G(i));
        jac.insert(ii, iik) = -lam;
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
        jac.insert(ii, iig) = -2.0 / h / h - lam * 2 * F(i);
        jac.insert(ii, iigp) = 1 / h / h - lam * H(i) / 2.0 / h;
        jac.insert(ii, iigm) = 1 / h / h + lam * H(i) / 2.0 / h;
        jac.insert(ii, iih) = -lam * (G(i + 1) - G(i - 1)) / 2.0 / h;
        jac.insert(ii, iif) = -lam * (2.0 * G(i));
        //std::cout << ii << std::endl;
    }

    //right boundary
    rhs(NN - 3) = F(N - 1) - bc[3];
    rhs(NN - 2) = G(N - 1) - bc[4];
    rhs(NN - 1) = H(N - 1) - bc[5];

    jac.insert(NN - 3, NN - 4) = 1;
    jac.insert(NN - 2, NN - 3) = 1;
    jac.insert(NN - 1, NN - 2) = 1;

    //std::cout << jac << std::endl;

    SparseLU<SparseMatrix<double>> solver;
    solver.compute(jac);

    if (solver.info() != Success)
    {
        throw std::invalid_argument("LU failed.");
    }

    VectorXd du = solver.solve(-rhs);
    double res = rhs.norm();

    std::pair<VectorXd, double> out = {du + u, res};
    return out;
}