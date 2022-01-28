#include <cmath>
#include <fstream>
#include <string>

#include <Eigen/Eigenvalues>

#include "rotst.hpp"

void rs_fun(int NSTEPS, int NPOINTS)
{

	//int n = 99;
	int n = NPOINTS;
	double l = 1.0;
	double h = l / (n + 1);
	int fields = 3;
	int size = fields * n + 1;

	Eigen::VectorXd G = Eigen::VectorXd::Zero(size);
	Eigen::MatrixXd J_u = Eigen::MatrixXd::Zero(size, size);
	Eigen::VectorXd J_lam = Eigen::VectorXd::Zero(size);

	//solution vectors
	Eigen::VectorXd u_0 = Eigen::VectorXd::Zero(size);
	Eigen::VectorXd u_dot = Eigen::VectorXd::Zero(size);
	double lam_dot = 1.0;
	double lam_0 = 625.0;
	double N = 0.0;
	double ds = 0.0;
	double kappa;

	std::array<double, 6> bc = {0.0, 0.0, 0.0, 0.0, 1.0, 0.0};

	//continuation LHS,RHS
	Eigen::MatrixXd LHS = Eigen::MatrixXd::Zero(size + 1, size + 1);
	Eigen::VectorXd RHS = Eigen::VectorXd::Zero(size + 1);
	std::pair<Eigen::MatrixXd, Eigen::VectorXd> assembly_set;

	//solve for 0th state
	double res = 1.0;
	int it = 0;
	std::cout << "===== Initial convergence Re=" << lam_0 << "=====" << std::endl;
	while (res > 1E-6 && it < 30)
	{

		J_u = eval_jacobian_u(u_0, lam_0, h, bc);
		J_lam = eval_jacobian_lam(u_0, lam_0, h, bc);
		G = eval_G(u_0, lam_0, h, bc);

		Eigen::VectorXd du = J_u.colPivHouseholderQr().solve(-G);
		u_0 += du;

		res = du.norm();
		std::cout << it << ": " << res << std::endl;

		it++;
	}

	//gradient
	assembly_set = assembly_vec(J_u, J_lam, u_dot, lam_dot);
	LHS = assembly_set.first;
	RHS = assembly_set.second;
	//	Eigen::VectorXd dx = LHS.colPivHouseholderQr().solve(RHS);
	Eigen::VectorXd dx = LHS.partialPivLu().solve(RHS);

	u_dot = dx(Eigen::seq(0, Eigen::placeholders::last - 1, 1));
	lam_dot = dx(Eigen::placeholders::last);

	//solution monitoring
	std::vector<double> R = {};
	std::vector<double> C = {};
	std::vector<double> m_ev = {};
	std::vector<double> k_array = {};
	std::vector<double> ds_arr = {};

	ds = -1.0;

	//continuation
	double kappa_aim = 1E-3;

	Eigen::VectorXd u_dot_prev;
	std::tuple<Eigen::VectorXd, Eigen::VectorXd, double, double, double> sol;

	for (int i = 0; i < NSTEPS; i++)
	{
		u_dot_prev = u_dot;
		sol = pc_iteration_k(u_0, u_dot, lam_0, lam_dot, h, bc, ds);
		R.push_back(lam_0);
		C.push_back(u_0(Eigen::placeholders::last));

		//unpack sol
		u_0 = std::get<0>(sol);
		u_dot = std::get<1>(sol);
		lam_0 = std::get<2>(sol);
		lam_dot = std::get<3>(sol);
		kappa = std::get<4>(sol);

		//step auto amendment
		ds = ds * std::sqrt(kappa_aim / kappa);
		std::cout << "------------- i = " << i << std::endl;
		//std::cout<<"k "<<u_0(Eigen::placeholders::last)<<std::endl;

		if (lam_0 < 0 || lam_0 > 625)
		{
			break;
		}
	}
}

//double max_real_ev

Eigen::MatrixXd eval_jacobian_u(Eigen::VectorXd u, double lam, double h, std::array<double, 6> bc)
{

	int fields = 3;
	int N = u.rows() - 1;
	double Re = lam;

	Eigen::MatrixXd J = Eigen::MatrixXd::Zero(u.rows(), u.rows());

	int last = N;
	Eigen::VectorXd B = u;

	double Fa = bc[0];
	double Ga = bc[1];
	double Ha = bc[2];
	double Fb = bc[3];
	double Gb = bc[4];
	double Hb = bc[5];

	for (int i = fields; i < N - fields; i += fields)
	{
		J(i, i) = -2 / h / h - Re * 2 * B(i);
		J(i, i + 1) = -Re * (-2 * B(i + 1));
		J(i, i + 2) = -Re * ((B(i + 3) - B(i - 3)) / 2.0 / h);
		J(i, i + 3) = 1 / h / h - Re * B(i + 2) / 2.0 / h;
		J(i, i - 3) = 1 / h / h - Re * (-B(i + 2) / 2.0 / h);
		J(i, last) = -Re;

		J(i + 1, i) = -Re * 2 * B(i + 1);
		J(i + 1, i + 1) = -2 / h / h - Re * 2 * B(i);
		J(i + 1, i + 2) = -Re * (B(i + 4) - B(i - 2)) / 2.0 / h;
		J(i + 1, i + 4) = 1 / h / h - Re * B(i + 2) / 2.0 / h;
		J(i + 1, i - 2) = 1 / h / h - Re * (-B(i + 2) / 2.0 / h);

		J(i + 2, i) = 1;
		J(i + 2, i + 3) = 1;
		J(i + 2, i + 5) = 1 / h;
		J(i + 2, i + 2) = -1 / h;
	}

	int i;
	//Boundaries
	i = 0;
	J(i, i) = -2 / h / h - Re * 2 * B(i);
	J(i, i + 1) = -Re * (-2 * B(i + 1));
	J(i, i + 2) = -Re * ((B(i + 3) - Fa) / 2 / h);
	J(i, i + 3) = 1 / h / h - Re * B(i + 2) / 2 / h;
	//J(i,i-3)=   1/h/h       -Re*(-B(i+2)/2/h);
	J(i, last) = -Re;

	J(i + 1, i) = -Re * 2 * B(i + 1);
	J(i + 1, i + 1) = -2 / h / h - Re * 2 * B(i);
	J(i + 1, i + 2) = -Re * (B(i + 4) - Ga) / 2 / h;
	J(i + 1, i + 4) = 1 / h / h - Re * B(i + 2) / 2 / h;
	//J(i+1,i-2)=1/h/h        -Re(-B(i+2)/2/h);

	J(i + 2, i) = 1;
	J(i + 2, i + 3) = 1;
	J(i + 2, i + 5) = 1 / h;
	J(i + 2, i + 2) = -1 / h;

	i = N - fields;
	J(i, i) = -2 / h / h - Re * 2 * B(i);
	J(i, i + 1) = -Re * (-2 * B(i + 1));
	J(i, i + 2) = -Re * ((Fb - B(i - 3)) / 2 / h);
	//J(i,i+3)=   1/h/h       -Re*B(i+2)/2/h;
	J(i, i - 3) = 1 / h / h - Re * (-B(i + 2) / 2 / h);
	J(i, last) = -Re;

	J(i + 1, i) = -Re * 2 * B(i + 1);
	J(i + 1, i + 1) = -2 / h / h - Re * 2 * B(i);
	J(i + 1, i + 2) = -Re * (Gb - B(i - 2)) / 2 / h;
	//J(i+1,i+4)=1/h/h        -Re*B(i+2)/2/h;
	J(i + 1, i - 2) = 1 / h / h - Re * (-B(i + 2) / 2 / h);

	J(i + 2, i) = 1;
	//J(i+2,i+3)=1;
	//J(i+2,i+5)=1/2/h;
	J(i + 2, i + 2) = -1 / h;

	J(last, 0) = 1;
	J(last, 2) = 1 / h;

	return J;
}

Eigen::VectorXd eval_G(Eigen::VectorXd u, double lam, double h, std::array<double, 6> bc)
{

	int fields = 3;
	int N = u.rows() - 1;
	double Re = lam;

	Eigen::VectorXd g = Eigen::VectorXd::Zero(u.rows());

	int last = N;
	Eigen::VectorXd B = u;

	double Fa = bc[0];
	double Ga = bc[1];
	double Ha = bc[2];
	double Fb = bc[3];
	double Gb = bc[4];
	double Hb = bc[5];

	double C = B[last];

	for (int i = fields; i < N - fields; i += fields)
	{
		g[i] = (B[i + 3] - 2 * B[i] + B[i - 3]) / h / h - Re * (B[i + 2] * (B[i + 3] - B[i - 3]) / 2 / h + B[i] * B[i] - B[i + 1] * B[i + 1] + C);
		g[i + 1] = (B[i + 4] - 2 * B[i + 1] + B[i - 2]) / h / h - Re * (B[i + 2] * (B[i + 4] - B[i - 2]) / 2 / h + 2 * B[i] * B[i + 1]);
		g[i + 2] = (B[i + 5] - B[i + 2]) / h + 2 * (B[i + 3] + B[i]) / 2;
	}

	int i;
	//Boundaries
	i = 0;
	g[i] = (B[i + 3] - 2 * B[i] + Fa) / h / h - Re * (B[i + 2] * (B[i + 3] - Fa) / 2 / h + B[i] * B[i] - B[i + 1] * B[i + 1] + C);
	g[i + 1] = (B[i + 4] - 2 * B[i + 1] + Ga) / h / h - Re * (B[i + 2] * (B[i + 4] - Ga) / 2 / h + 2 * B[i] * B[i + 1]);
	g[i + 2] = (B[i + 5] - B[i + 2]) / h + 2 * (B[i] + B[i + 3]) / 2;

	i = N - fields;
	g[i] = (Fb - 2 * B[i] + B[i - 3]) / h / h - Re * (B[i + 2] * (Fb - B[i - 3]) / 2 / h + B[i] * B[i] - B[i + 1] * B[i + 1] + C);
	g[i + 1] = (Gb - 2 * B[i + 1] + B[i - 2]) / h / h - Re * (B[i + 2] * (Gb - B[i - 2]) / 2 / h + 2 * B[i] * B[i + 1]);
	g[i + 2] = (Hb - B[i + 2]) / h + 2 * (B[i] + Fb) / 2;

	g[last] = (B[2] - Ha) / h + 2 * (B[0] + Fa) / 2;
	return g;
}

Eigen::VectorXd eval_jacobian_lam(Eigen::VectorXd u, double lam, double h, std::array<double, 6> bc)
{

	int fields = 3;
	int N = u.rows() - 1;
	double Re = lam;

	Eigen::VectorXd J_lam = Eigen::VectorXd::Zero(u.rows());

	int last = N;
	Eigen::VectorXd B = u;

	double Fa = bc[0];
	double Ga = bc[1];
	double Ha = bc[2];
	double Fb = bc[3];
	double Gb = bc[4];
	double Hb = bc[5];

	double C = B[last];

	for (int i = fields; i < N - fields; i += fields)
	{

		J_lam[i] = -(B[i + 2] * (B[i + 3] - B[i - 3]) / 2 / h + B[i] * B[i] - B[i + 1] * B[i + 1] + C);
		J_lam[i + 1] = -(B[i + 2] * (B[i + 4] - B[i - 2]) / 2 / h + 2 * B[i] * B[i + 1]);
		J_lam[i + 2] = 0;
	}

	int i;
	//Boundaries
	i = 0;

	J_lam[i] = -(B[i + 2] * (B[i + 3] - Fa) / 2 / h + B[i] * B[i] - B[i + 1] * B[i + 1] + C);
	J_lam[i + 1] = -(B[i + 2] * (B[i + 4] - Ga) / 2 / h + 2 * B[i] * B[i + 1]);
	J_lam[i + 2] = 0;

	i = N - fields;

	J_lam[i] = -(B[i + 2] * (Fb - B[i - 3]) / 2 / h + B[i] * B[i] - B[i + 1] * B[i + 1] + C);
	J_lam[i + 1] = -(B[i + 2] * (Gb - B[i - 2]) / 2 / h + 2 * B[i] * B[i + 1]);
	J_lam[i + 2] = 0;

	J_lam[last] = 0;

	return J_lam;
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd> assembly_vec(Eigen::MatrixXd J_u, Eigen::VectorXd J_lam, Eigen::VectorXd u_dot, double lam_dot)
{
	int size = u_dot.rows();
	int n = size + 1;
	Eigen::MatrixXd LHS = Eigen::MatrixXd::Zero(n, n);
	Eigen::VectorXd RHS = Eigen::VectorXd::Zero(n);

	for (int i = 0; i < n - 1; i++)
	{
		for (int j = 0; j < n - 1; j++)
		{
			LHS(i, j) = J_u(i, j);
		}
		LHS(i, n - 1) = J_lam(i);
	}

	RHS(n - 1) = 1.0;

	for (int j = 0; j < n - 1; j++)
	{
		LHS(n - 1, j) = u_dot(j);
	}
	LHS(n - 1, n - 1) = lam_dot;

	std::pair<Eigen::MatrixXd, Eigen::VectorXd> assembly_set(LHS, RHS);
	return assembly_set;
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd, double, double, double> pc_iteration_k(Eigen::VectorXd u_0, Eigen::VectorXd u_dot, double lam_0, double lam_dot, double h, std::array<double, 6> bc, double ds)
{
	//predictor
	Eigen::VectorXd u = u_0 + ds * u_dot;
	double lam = lam_0 + ds * lam_dot;
	std::cout << "===== Re=" << lam << "=====" << std::endl;

	//corrector
	double res = 1;
	int it = 0;

	//for kappa
	Eigen::VectorXd du0 = u;
	Eigen::VectorXd du1 = u;

	int size = u.rows();
	Eigen::MatrixXd J_u;
	Eigen::VectorXd J_lam;
	Eigen::VectorXd G;
	double N;

	while ((res > 1E-6 && it < 25) || it == 1)
	{
		J_u = eval_jacobian_u(u, lam, h, bc);
		J_lam = eval_jacobian_lam(u, lam, h, bc);
		G = eval_G(u, lam, h, bc);
		N = eval_N(u, lam, u_0, lam_0, u_dot, lam_dot, ds);

		//master matrix
		Eigen::MatrixXd LHS = Eigen::MatrixXd::Zero(size + 1, size + 1);
		Eigen::VectorXd RHS = Eigen::VectorXd::Zero(size + 1);
		std::pair<Eigen::MatrixXd, Eigen::VectorXd> assembly_set;

		assembly_set = assembly(J_u, J_lam, u_dot, lam_dot, G, N);
		LHS = assembly_set.first;
		RHS = assembly_set.second;
		Eigen::VectorXd dx = LHS.colPivHouseholderQr().solve(RHS);

		Eigen::VectorXd du = dx(Eigen::seq(0, Eigen::placeholders::last - 1, 1));
		double dlam = dx(Eigen::placeholders::last);

		u += du;
		lam += dlam;

		res = dx.norm();
		std::cout << it << ":" << res << std::endl;

		if (it == 0)
		{
			du0 = du;
		}
		else if (it == 1)
		{
			du1 = du;
		}
		it += 1;
	}

	double kappa = du1.norm() / du0.norm();

	//gradient
	std::pair<Eigen::MatrixXd, Eigen::VectorXd> assembly_set;
	assembly_set = assembly_vec(J_u, J_lam, u_dot, lam_dot);
	Eigen::MatrixXd LHS = assembly_set.first;
	Eigen::VectorXd RHS = assembly_set.second;
	Eigen::VectorXd dx = LHS.colPivHouseholderQr().solve(RHS);

	u_dot = dx(Eigen::seq(0, Eigen::placeholders::last - 1, 1));
	lam_dot = dx(Eigen::placeholders::last);

	//increment
	u_0 = u;
	lam_0 = lam;
	std::cout << "lam_dot= " << lam_dot << std::endl;
	std::tuple<Eigen::VectorXd, Eigen::VectorXd, double, double, double> sol = {u_0, u_dot, lam_0, lam_dot, kappa};
	return sol;
}

double eval_N(Eigen::VectorXd u, double lam, Eigen::VectorXd u_0, double lam_0, Eigen::VectorXd u_dot, double lam_dot, double ds)
{
	double N = (u - u_0).dot(u_dot) + (lam - lam_0) * lam_dot - ds;
	return N;
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd> assembly(Eigen::MatrixXd J_u, Eigen::VectorXd J_lam, Eigen::VectorXd u_dot, double lam_dot, Eigen::VectorXd G, double N)
{
	int size = u_dot.rows();
	int n = size + 1;
	Eigen::MatrixXd LHS = Eigen::MatrixXd::Zero(n, n);
	Eigen::VectorXd RHS = Eigen::VectorXd::Zero(n);

	for (int i = 0; i < n - 1; i++)
	{
		for (int j = 0; j < n - 1; j++)
		{
			LHS(i, j) = J_u(i, j);
		}
		LHS(i, n - 1) = J_lam(i);
		RHS(i) = -G(i);
	}
	RHS(n - 1) = -N;

	for (int j = 0; j < n - 1; j++)
	{
		LHS(n - 1, j) = u_dot(j);
	}
	LHS(n - 1, n - 1) = lam_dot;

	std::pair<Eigen::MatrixXd, Eigen::VectorXd> assembly_set(LHS, RHS);
	return assembly_set;
}

// other function
void help_fun()
{
	std::cout << "This is rotst program." << std::endl;

	std::ifstream infile(".helpFile");

	std::string str;

	while(!infile.eof())
	{
		std::getline(infile,str);
		std::cout<<str<<std::endl;
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

	//computation of evs
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
		//std::cout << s << c1 << ii << c2 << std::endl;
		
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
