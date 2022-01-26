#include "rotst.hpp"


void help_fun(){
	std::cout<<"This is rotst program."<<std::endl;
}

void ver_fun(){
        std::cout<<"Version: 1.0"<<std::endl;
}

void lsol_fun(){
        std::cout<<"=====Linear solve using eigen====="<<std::endl;
	int n;
	std::cout<<"Dimension:"<<std::endl;
	std::cin>>n;

	Eigen::MatrixXd A(n,n);
	Eigen::VectorXd b(n);

	std::cout<<"A matrix:"<<std::endl;
	double val;
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){

			std::cin>>val;
			A(i,j)=val;
		}
	}
	std::cout<<"A matrix:"<<std::endl;
	std::cout<<A<<std::endl;

	std::cout<<"b vec:"<<std::endl;
	for(int i=0; i<n; i++){
                std::cin>>val;
                b(i)=val;
        }
        std::cout<<"b vec:"<<std::endl;
        std::cout<<b<<std::endl;

	Eigen::VectorXd sol=A.colPivHouseholderQr().solve(b);
        std::cout<<"Solution:"<<std::endl;
        std::cout<<sol<<std::endl;


}


