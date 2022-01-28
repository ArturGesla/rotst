#include <iostream>
#include <string>
#include <tuple>
#include <vector>

void help_fun();
void ver_fun();
void lsol_fun();
void ev_fun();
void rs_fun(int NSTEPS, int NPOINTS);

std::vector<double> readInput();

int main(int argc, char **argv)
{
	std::vector<double> data = readInput();

	int NSTEPS = data[0];
	int NPOINTS = data[1];

	//	std::cout << "Have " << argc << " arguments:" << std::endl;
	//	for (int i = 0; i < argc; ++i) {
	//        	std::cout << argv[i] << std::endl;
	//    	}

	std::string help_msg = "--help";
	std::string ver_msg = "--version";
	std::string lsol_msg = "lsol";
	std::string ev_msg = "ev";
	std::string rs_msg = "-rs";

	for (int i = 0; i < argc; i++)
	{
		if (argv[i] == help_msg)
			help_fun();
		if (argv[i] == ver_msg)
			ver_fun();
		if (argv[i] == lsol_msg)
			lsol_fun();
		if (argv[i] == ev_msg)
			ev_fun();
		if (argv[i] == rs_msg)
			rs_fun(NSTEPS, NPOINTS);
	}
	return 0;
}
