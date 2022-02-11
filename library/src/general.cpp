#include "general.hpp"



void help_fun()
{
    std::cout << "This is rotst program." << std::endl;

    std::ifstream infile(".helpFile");

    std::string str;

    while (!infile.eof())
    {
        std::getline(infile, str);
        std::cout << str << std::endl;
    }
}

void ver_fun()
{
    std::cout << "Version: 1.0" << std::endl;
}

void lsol_fun()
{
    std::cout << "=====Linear solve using eigen=====" << std::endl;
    int n;
    std::cout << "Dimension:" << std::endl;
    std::cin >> n;

    Eigen::MatrixXd A(n, n);
    Eigen::VectorXd b(n);

    std::cout << "A matrix:" << std::endl;
    double val;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {

            std::cin >> val;
            A(i, j) = val;
        }
    }
    std::cout << "A matrix:" << std::endl;
    std::cout << A << std::endl;

    std::cout << "b vec:" << std::endl;
    for (int i = 0; i < n; i++)
    {
        std::cin >> val;
        b(i) = val;
    }
    std::cout << "b vec:" << std::endl;
    std::cout << b << std::endl;

    Eigen::VectorXd sol = A.colPivHouseholderQr().solve(b);
    std::cout << "Solution:" << std::endl;
    std::cout << sol << std::endl;
}

void ev_fun()
{
    std::cout << "=====EV search using eigen=====" << std::endl;
    int n;
    std::cout << "Dimension:" << std::endl;
    std::cin >> n;

    Eigen::MatrixXd A(n, n);
    Eigen::MatrixXd B(n, n);

    std::cout << "A matrix:" << std::endl;
    double val;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {

            std::cin >> val;
            A(i, j) = val;
        }
    }
    std::cout << "A matrix:" << std::endl;
    std::cout << A << std::endl;

    std::cout << "B matrix:" << std::endl;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {

            std::cin >> val;
            B(i, j) = val;
        }
    }
    std::cout << "B matrix:" << std::endl;
    std::cout << B << std::endl;

    // computation of evs
    Eigen::GeneralizedEigenSolver<Eigen::MatrixXd> ges;
    ges.compute(A, B);

    Eigen::VectorXd alpha_real = Eigen::VectorXd(ges.alphas().real());
    Eigen::VectorXd alpha_imag = Eigen::VectorXd(ges.alphas().imag());
    Eigen::VectorXd beta = Eigen::VectorXd(ges.betas());

    for (int i = 0; i < n; i++)
    {
        std::cout << "Real: " << alpha_real(i) / beta(i) << "	Imag: " << alpha_imag(i) / beta(i) << std::endl;
    }

    std::cout << "Aplhas:" << std::endl
              << ges.alphas() << std::endl;

    std::cout << "Betas:" << std::endl
              << ges.betas() << std::endl;
}

std::vector<double> readInput()
{
    std::vector<double> data(10);

    std::ifstream infile("inDATA");
    std::string s;
    char c1, c2;
    double ii;

    data[0] = 1000;
    data[1] = 99;

    while (infile >> s >> c1 >> ii >> c2)
    {
        // std::cout << s << c1 << ii << c2 << std::endl;

        if (s == "NSTEPS")
        {
            data[0] = ii;
        }
        if (s == "NPOINTS")
        {
            data[1] = ii;
        }
        if (s == "PAR")
        {
            data[3] = ii;
        }
    }

    return data;
}

// std::vector<double> data = readInput();

// 	int NSTEPS = data[0];
// 	int NPOINTS = data[1];
