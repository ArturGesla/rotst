#include "diff2D.hpp"

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
