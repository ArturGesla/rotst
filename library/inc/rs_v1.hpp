#include <iostream>
#include <Eigen/Dense>
#include <vector>

void rs_v1(int NSTEPS=10, int NPOINTS=99);

Eigen::MatrixXd eval_jacobian_u(Eigen::VectorXd u, double lam, double h, std::array<double,6> bc);
Eigen::VectorXd eval_jacobian_lam(Eigen::VectorXd u_0, double lam_0, double h, std::array<double,6> bc);
Eigen::VectorXd eval_G(Eigen::VectorXd u_0, double lam_0, double h, std::array<double,6> bc);
double eval_N(Eigen::VectorXd u, double lam, Eigen::VectorXd u_0, double lam_0, Eigen::VectorXd u_dot, double lam_dot, double ds);

std::pair<Eigen::MatrixXd,Eigen::VectorXd> assembly_vec(Eigen::MatrixXd J_u, Eigen::VectorXd J_lam, Eigen::VectorXd u_dot, double lam_dot);
std::pair<Eigen::MatrixXd, Eigen::VectorXd> assembly(Eigen::MatrixXd J_u, Eigen::VectorXd J_lam, Eigen::VectorXd u_dot, double lam_dot, Eigen::VectorXd G, double N);

std::tuple<Eigen::VectorXd,Eigen::VectorXd,double,double,double> pc_iteration_k(Eigen::VectorXd u_0, Eigen::VectorXd u_dot,double lam_0, double lam_dot, double h, std::array<double,6> bc,double ds);

