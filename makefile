INC=-I../eigen

rotst: main.cpp rotst.cpp 
	g++ $(INC) -o rotst main.cpp rotst.cpp -I.

clean:
	rm rotst

o3: main.cpp rotst.cpp
	rm rotst
	g++ $(INC) -O3 -o rotst main.cpp rotst.cpp -I.
        

