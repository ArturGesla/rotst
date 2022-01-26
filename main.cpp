#include <iostream>
#include <string>

#include "rotst.hpp"


int main(int argc, char** argv) {

//	std::cout << "Have " << argc << " arguments:" << std::endl;
//	for (int i = 0; i < argc; ++i) {
//        	std::cout << argv[i] << std::endl;
//    	}

	std::string help_msg=	"--help";
	std::string ver_msg=	"--version";
	std::string lsol_msg=	"lsol";

	for (int i=0; i<argc; i++){
		if(argv[i]==help_msg) 	help_fun();
		if(argv[i]==ver_msg) 	ver_fun();
		if(argv[i]==lsol_msg) 	lsol_fun();
	}
	return 0;
}
