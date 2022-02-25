#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "general.hpp"
#include "rs1D.hpp"
#include "rs2D.hpp"
#include "rs2Dfv.hpp"
#include "condiff1D.hpp"
#include "diff2D.hpp"


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

			double lx=10;
			double lz=1;

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

			if (i < argc - 1 && argv[i + 1] == std::string("-lx"))
			{
				lx = std::stof(argv[i + 2]);
				i += 2;
			}

			if (i < argc - 1 && argv[i + 1] == std::string("-lz"))
			{
				lz = std::stof(argv[i + 2]);
				i += 2;
			}

			if (verbose)
				std::cout << "RE: " << re << " nx: " << nx << " nz: " << nz << " nnewt: " << nnewt << " nconti: " << nconti <<" lx: "<<lx <<" lz: "<<lz<<std::endl;

			rs2Dfv::rs2D(re, nx, nz, nnewt, nconti, lx,lz);
		}
	}
	return 0;
}
