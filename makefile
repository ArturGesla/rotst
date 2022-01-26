INC=-I../eigen

rotst: main.cpp
	g++ $(INC) -o rotst main.cpp -I.
