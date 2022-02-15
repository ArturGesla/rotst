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
	rs2D();
}

void execute(std::string msg)
{
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
	if (msg == test_msg)
		test();
}

int main(int argc, char **argv)
{

	for (int i = 0; i < argc; i++)
	{
		execute(argv[i]);
	}
	return 0;
}
