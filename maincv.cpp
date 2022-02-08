#include <iostream>
#include <fstream>

#include <Eigen/Dense>
#include <Eigen/Sparse>

int main()
{

    using namespace Eigen;
    using namespace std;

    // 1D conv diff
    int N = 11;
    double h = 1.0 / (N - 1);
    N += 2.0; // plus 2 ghosts

    double D = 1;
    double U = 25;

    // bc
    double phia = 0.0;
    double phib = 1.0;

    VectorXd phi = VectorXd::Ones(N);
    VectorXd rhs = VectorXd::Zero(N);

    SparseMatrix<double> jac(N, N);

    // left boundary
    {
        int ii = 0;
        rhs(ii) = phi(ii +1) - phia;
        jac.insert(ii, ii +1) = 1;
        //rhs(ii) = (phi(ii + 2) - phi(ii)) / h / 2.0 - 2.0;
        //jac.insert(ii, ii + 2) = 1.0 / 2.0 / h;
        //jac.insert(ii, ii) = -1.0 / 2.0 / h;
    }
    for (size_t i = 0; i < N - 2; i++)
    {
        int ii = i + 1;
        double diff = D * (phi(ii + 1) - 2 * phi(ii) + phi(ii - 1)) / h / h;
        double conv = U * (phi(ii + 1) - phi(ii - 1)) / 2.0 / h;
        rhs(ii) = -conv + diff;
        jac.insert(ii, ii + 1) = D * 1.0 / h / h - U / 2.0 / h;
        jac.insert(ii, ii) = -2.0 * D / h / h;
        jac.insert(ii, ii - 1) = D * 1.0 / h / h + U / 2.0 / h;
    }

    // right boundary
    {
        int ii = N - 1;
        rhs(ii) = phi(ii - 1) - phib;
        jac.insert(ii, ii - 1) = 1;
    }

    std::cout << rhs << std::endl;
    std::cout << jac << std::endl;

    SparseLU<SparseMatrix<double>> solver;
    solver.compute(jac);

    if (solver.info() != Success)
    {
        throw; // std::cout<<"LU failed"<<std::endl;
    }

    VectorXd du = solver.solve(-rhs);
    phi += du;

    std::cout << phi << std::endl;

    std::cout << "Pe = " << U * h / D << std::endl;

    ofstream myfile;
    myfile.open("res-cd");
    myfile << phi;
    myfile.close();
}