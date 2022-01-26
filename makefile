INC=-I../eigen

rotst: main.cpp rotst.cpp 
	g++ $(INC) -o rotst main.cpp rotst.cpp -I.
