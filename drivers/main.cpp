#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "general.hpp"
#include "rs1D.hpp"
#include "rs2D.hpp"
#include "condiff1D.hpp"
#include "diff2D.hpp"

void test()
{
	//rs2D();
}

int main(int argc, char **argv)
{
	bool verbose = false;

	for (int i = 0; i < argc; i++)
	{
		std::string msg = argv[i];

		std::string help_msg = "--help";
		std::string ver_msg = "--version";
		std::string lsol_msg = "lsol";
		std::string ev_msg = "ev";
		std::string test_msg = "-t";

		if (msg == help_msg)
			help_fun();
		if (msg == ver_msg)
			ver_fun();
		if (msg == lsol_msg)
			lsol_fun();
		if (msg == ev_msg)
			ev_fun();
		if (msg == "-v")
			verbose = true;
		if (msg == test_msg)
		{
			double re = 1;
			size_t nx = 30;
			size_t nz = 30;

			size_t nnewt = 5;
			size_t nconti = 1;

			if (i < argc - 1 && argv[i + 1] == std::string("-re"))
			{
				re = std::stof(argv[i + 2]);
				i += 2;

				//std::cout << re << std::endl;
			}

			if (i < argc - 1 && argv[i + 1] == std::string("-nx"))
			{
				nx = std::stof(argv[i + 2]);
				i += 2;
			}
			if (i < argc - 1 && argv[i + 1] == std::string("-nz"))
			{
				nz = std::stof(argv[i + 2]);
				i += 2;
			}

			if (i < argc - 1 && argv[i + 1] == std::string("-nnewt"))
			{
				nnewt = std::stof(argv[i + 2]);
				i += 2;
			}

			if (i < argc - 1 && argv[i + 1] == std::string("-nconti"))
			{
				nconti = std::stof(argv[i + 2]);
				i += 2;
			}

			if (verbose)
				std::cout << "RE: " << re << " nx: " << nx << " nz: " << nz << " nnewt: " << nnewt << " nconti: " << nconti << std::endl;

			rs2D(re, nx, nz, nnewt, nconti);
		}
	}
	return 0;
}
